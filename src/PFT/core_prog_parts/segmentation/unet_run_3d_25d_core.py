"""Z10/Z12 merged-RGB 2.5D U-Net inference and full-stack mask creation.

The network is evaluated only at the two annotated target positions, Z10 and
Z12. Each prediction uses the wavelength-mapped merged RGB context around its
middle slice. The two binary masks are combined by pixelwise maximum, which is
equivalent to a logical union for 0/1 masks. The combined two-dimensional mask
is then broadcast to every Z-slice and saved as a ZYX OME-Zarr semantic mask.

No fluorescence intensities are modified in this module. Background attenuation
is performed later by ``mask_application_3d.py`` on the saved deconvolution
output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

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
    DEFAULT_TRAINING_SLICES_1BASED,
    annotation_sample_dir,
    find_slice_mask,
    relative_volume_path,
    volume_key,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (
    bce_dice_loss,
    dice_coef,
    iou_coef,
)
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import (
    UNet25DTrainConfig,
    make_merged_rgb_context_slice,
    open_3d_image_czyx,
    predict_25d_probability,
    read_binary_slice_mask,
    resolve_rgb_source_channels,
)


@dataclass
class UNet25DRunConfig:
    """Configuration for Z10/Z12 mask-only merged-RGB inference."""

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
    inference_slices_1based: tuple[int, ...] = DEFAULT_TRAINING_SLICES_1BASED
    preview_slices_1based: tuple[int, ...] = DEFAULT_TRAINING_SLICES_1BASED
    save_probability: bool = True
    combination_mode: str = "pixelwise_max"


@dataclass(frozen=True)
class UNet25DRunOutput:
    """Saved products from one combined-mask inference run."""

    sample: str
    source_zarr: Path
    mask_zarr: Path
    probability_zarr: Path | None
    combined_mask_tif: Path
    source_slice_dir: Path
    preview_dir: Path
    report_json: Path


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
    """Require the training summary produced by the updated merged-RGB code."""
    summary_path = model_path.parent / "u_net_3d_25d_training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Model contract summary is missing: {summary_path}. Retrain the model with the "
            "updated Z10/Z12 merged-RGB training script before inference."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_contract = "pft_3d_25d_merged_rgb_z10_z12_v1"
    if summary.get("model_contract") != expected_contract:
        raise ValueError(
            f"Model summary contract is {summary.get('model_contract')!r}, expected "
            f"{expected_contract!r}. Retrain with the updated workflow."
        )
    if tuple(summary.get("training_slices_1based") or ()) != DEFAULT_TRAINING_SLICES_1BASED:
        raise ValueError(
            f"Model was not trained on Z10/Z12: {summary.get('training_slices_1based')}"
        )
    if int(summary.get("model_input_channels", -1)) != 9:
        raise ValueError(
            f"Model summary does not describe a nine-channel merged-RGB input: "
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


def _validate_source_slices(
    slices_1based: Sequence[int],
    z_count: int,
) -> tuple[int, int]:
    slices = tuple(sorted({int(value) for value in slices_1based}))
    if slices != DEFAULT_TRAINING_SLICES_1BASED:
        raise ValueError(
            f"This workflow is fixed to Z10 and Z12 inference; received {slices}."
        )
    for value in slices:
        if value < 2 or value > z_count - 1:
            raise ValueError(
                f"Z{value} lacks complete Z-1/Z/Z+1 context for a stack with Z={z_count}."
            )
    return slices[0], slices[1]


def combine_source_masks_pixelwise_max(
    masks_by_slice: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Combine binary source-slice masks by elementwise maximum.

    For 0/1 masks this is equivalent to a logical union: a foreground pixel in
    either Z10 or Z12 remains foreground in the combined mask.
    """
    if not masks_by_slice:
        raise ValueError("At least one source-slice mask is required")
    ordered = [np.asarray(masks_by_slice[key]) for key in sorted(masks_by_slice)]
    reference_shape = ordered[0].shape
    for mask in ordered:
        if mask.ndim != 2 or mask.shape != reference_shape:
            raise ValueError(
                f"Source masks must be matching 2D arrays; expected {reference_shape}, got {mask.shape}"
            )
        values = np.unique(mask)
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError(f"Source masks must be binary 0/1; values={values[:20]}")
    return np.maximum.reduce(ordered).astype(np.uint8)


def combine_source_probabilities_pixelwise_max(
    probabilities_by_slice: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Combine source-slice probability maps by elementwise maximum."""
    if not probabilities_by_slice:
        raise ValueError("At least one source-slice probability map is required")
    ordered = [np.asarray(probabilities_by_slice[key], dtype=np.float32) for key in sorted(probabilities_by_slice)]
    reference_shape = ordered[0].shape
    for probability in ordered:
        if probability.ndim != 2 or probability.shape != reference_shape:
            raise ValueError(
                f"Probability maps must be matching 2D arrays; expected {reference_shape}, got {probability.shape}"
            )
        if not np.isfinite(probability).all():
            raise ValueError("Probability maps contain non-finite values")
    return np.maximum.reduce(ordered).astype(np.float32)


def _save_source_slice_products(
    *,
    source_slice_dir: Path,
    probabilities: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
    combined_probability: np.ndarray,
    combined_mask: np.ndarray,
) -> Path:
    """Save individual Z10/Z12 products and their combined 2D products."""
    source_slice_dir.mkdir(parents=True, exist_ok=True)
    for slice_number in sorted(probabilities):
        tiff.imwrite(
            source_slice_dir / f"z{slice_number:03d}_foreground_probability.tif",
            probabilities[slice_number].astype(np.float32),
        )
        tiff.imwrite(
            source_slice_dir / f"z{slice_number:03d}_pred_mask.tif",
            masks[slice_number].astype(np.uint8),
        )
    tiff.imwrite(
        source_slice_dir / "combined_probability_pixelwise_max.tif",
        combined_probability.astype(np.float32),
    )
    combined_mask_tif = source_slice_dir / "combined_mask_pixelwise_max.tif"
    tiff.imwrite(combined_mask_tif, combined_mask.astype(np.uint8))
    return combined_mask_tif


def _save_source_mask_previews(
    *,
    image_czyx,
    training_cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
    probabilities: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
    combined_mask: np.ndarray,
    sample: str,
    output_dir: Path,
    manual_mask_dir: Path | None,
) -> list[Path]:
    """Save Z10/Z12 prediction QC and one combined-mask QC figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    normalization_cache: dict[tuple[int, int], tuple[float, float]] = {}
    safe_sample = sample.replace("/", "__").replace("\\", "__")

    merged_images: dict[int, np.ndarray] = {}
    for slice_number in sorted(probabilities):
        z_index = slice_number - 1
        context = make_merged_rgb_context_slice(
            image_czyx,
            z_index,
            training_cfg,
            rgb_source_channels,
            normalization_cache=normalization_cache,
        )
        merged_images[slice_number] = context[1]
        reference = None
        if manual_mask_dir is not None:
            reference_path = find_slice_mask(manual_mask_dir, slice_number)
            if reference_path is not None:
                reference = read_binary_slice_mask(reference_path, masks[slice_number].shape)

        columns = 5 if reference is not None else 4
        figure, axes = plt.subplots(
            1, columns, figsize=(4 * columns, 4), constrained_layout=True
        )
        axes[0].imshow(merged_images[slice_number])
        axes[0].set_title(f"Merged RGB Z{slice_number}")
        axes[1].imshow(probabilities[slice_number], cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("Foreground probability")
        axes[2].imshow(masks[slice_number], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Predicted mask (0.5)")
        axes[3].imshow(merged_images[slice_number])
        axes[3].contour(masks[slice_number], levels=[0.5], colors="white", linewidths=0.7)
        axes[3].set_title("Predicted-mask overlay")
        if reference is not None:
            axes[4].imshow(reference, cmap="gray", vmin=0, vmax=1)
            axes[4].contour(masks[slice_number], levels=[0.5], colors="red", linewidths=0.6)
            axes[4].set_title("Manual mask + prediction")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{sample} | source inference Z{slice_number:03d}")
        path = output_dir / f"{safe_sample}__z{slice_number:03d}__predicted_mask_qc.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved.append(path)

    first, second = sorted(masks)
    figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    axes[0].imshow(masks[first], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Z{first} predicted mask")
    axes[1].imshow(masks[second], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Z{second} predicted mask")
    axes[2].imshow(combined_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Combined pixelwise maximum")
    axes[3].imshow(merged_images[first])
    axes[3].contour(combined_mask, levels=[0.5], colors="white", linewidths=0.7)
    axes[3].set_title(f"Combined mask on Z{first}")
    axes[4].imshow(merged_images[second])
    axes[4].contour(combined_mask, levels=[0.5], colors="white", linewidths=0.7)
    axes[4].set_title(f"Combined mask on Z{second}")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{sample} | Z{first}/Z{second} mask combination")
    combined_path = output_dir / f"{safe_sample}__combined_z{first:03d}_z{second:03d}_mask_qc.png"
    figure.savefig(combined_path, dpi=160)
    plt.close(figure)
    saved.append(combined_path)
    return saved


def run_3d_25d_unet_masks(cfg: UNet25DRunConfig) -> UNet25DRunOutput:
    """Infer Z10/Z12, combine their masks, and broadcast to the full stack."""
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
    if cfg.combination_mode != "pixelwise_max":
        raise ValueError("The agreed Z10/Z12 combination mode is pixelwise_max")
    if cfg.channels is not None:
        raise ValueError(
            "Merged-RGB inference always uses all three wavelength-mapped channels."
        )

    image = open_3d_image_czyx(input_zarr, level=cfg.level)
    if int(image.shape[0]) != 3:
        raise ValueError(f"Merged RGB inference requires C=3, received C={image.shape[0]}")
    source_slices = _validate_source_slices(
        cfg.inference_slices_1based, int(image.shape[1])
    )
    relative_volume = relative_volume_path(input_zarr)
    sample = volume_key(input_zarr)
    model_summary_path = _validate_model_contract(model_path)
    model = load_25d_model(model_path)
    expected_channels = int(model.input_shape[-1])
    if expected_channels != 9:
        raise ValueError(
            f"Merged-RGB model must expect 9 channels, but model input is {expected_channels}. "
            "Retrain with the updated Z10/Z12 merged-RGB training code."
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
        training_slices_1based=DEFAULT_TRAINING_SLICES_1BASED,
    )
    normalization_cache: dict[tuple[int, int], tuple[float, float]] = {}
    probabilities: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    for slice_number in source_slices:
        z_index = slice_number - 1
        print(
            f"[U-NET MASK] {sample}: infer Z{slice_number:03d} from "
            f"Z{slice_number - 1}/Z{slice_number}/Z{slice_number + 1} merged RGB"
        )
        probability = predict_25d_probability(
            model,
            image,
            z_index,
            training_cfg,
            rgb_source_channels,
            normalization_cache=normalization_cache,
        )
        probabilities[slice_number] = probability
        masks[slice_number] = (probability >= cfg.threshold).astype(np.uint8)

    combined_probability = combine_source_probabilities_pixelwise_max(probabilities)
    combined_mask = combine_source_masks_pixelwise_max(masks)
    if not np.any(combined_mask) or np.all(combined_mask):
        raise ValueError(
            "Combined Z10/Z12 mask must contain both foreground and background."
        )

    z_count = int(image.shape[1])
    mask_zyx = np.broadcast_to(
        combined_mask[None, ...],
        (z_count, *combined_mask.shape),
    ).copy()

    sample_dir = ensure_dir(output_root / relative_volume)
    source_slice_dir = ensure_dir(sample_dir / "source_slice_predictions")
    combined_mask_tif = _save_source_slice_products(
        source_slice_dir=source_slice_dir,
        probabilities=probabilities,
        masks=masks,
        combined_probability=combined_probability,
        combined_mask=combined_mask,
    )

    retained = copyable_root_metadata(input_zarr)
    source_channel_names = retained.pop("channel_names", None)
    source_omero = retained.pop("omero", None)
    processing = {
        "operation": "unet_3d_25d_z10_z12_combined_foreground_mask",
        "source_omezarr": str(input_zarr),
        "source_level": int(cfg.level),
        "model_path": str(model_path),
        "model_summary_path": str(model_summary_path),
        "model_contract": "pft_3d_25d_merged_rgb_z10_z12_v1",
        "inference_slices_1based": list(source_slices),
        "inference_contexts_1based": {
            str(value): [value - 1, value, value + 1] for value in source_slices
        },
        "input_representation": "wavelength-mapped merged RGB at each context Z",
        "rgb_source_channels": {
            "red": int(rgb_source_channels[0]),
            "green": int(rgb_source_channels[1]),
            "blue": int(rgb_source_channels[2]),
        },
        "rgb_wavelengths_nm": {"red": 561.0, "green": 488.0, "blue": 405.0},
        "combination_mode": "pixelwise_max",
        "combination_semantics": "logical union of thresholded Z10 and Z12 masks",
        "combined_mask_broadcast_to_all_z": True,
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
            "operation": "unet_3d_25d_z10_z12_combined_probability_2d",
            "stored_axes": "yx",
            "broadcast_to_z": False,
        }
        probability_zarr = save_ome_zarr(
            sample_dir / "combined_foreground_probability.ome.zarr",
            combined_probability,
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
    preview_dir = sample_dir / "qc_source_slices_and_combination"
    previews = _save_source_mask_previews(
        image_czyx=image,
        training_cfg=training_cfg,
        rgb_source_channels=rgb_source_channels,
        probabilities=probabilities,
        masks=masks,
        combined_mask=combined_mask,
        sample=sample,
        output_dir=preview_dir,
        manual_mask_dir=manual_mask_dir,
    )

    individual_foreground = {
        str(value): int(np.count_nonzero(masks[value])) for value in source_slices
    }
    report = {
        "sample": sample,
        "source_omezarr": str(input_zarr),
        "source_level": cfg.level,
        "model_path": str(model_path),
        "model_summary_path": str(model_summary_path),
        "mask_zarr": str(mask_zarr),
        "probability_zarr": str(probability_zarr) if probability_zarr else None,
        "combined_mask_tif": str(combined_mask_tif),
        "source_slice_dir": str(source_slice_dir),
        "inference_slices_1based": list(source_slices),
        "individual_foreground_pixels": individual_foreground,
        "combined_foreground_pixels": int(np.count_nonzero(combined_mask)),
        "combined_background_pixels": int(combined_mask.size - np.count_nonzero(combined_mask)),
        "combination_mode": "pixelwise_max",
        "full_mask_shape_zyx": list(mask_zyx.shape),
        "full_mask_dtype": str(mask_zyx.dtype),
        "broadcast_to_all_z": True,
        "threshold": 0.5,
        "intensity_modification_performed": False,
        "preview_files": [str(path) for path in previews],
        "configuration": asdict(cfg),
    }
    report_json = sample_dir / "u_net_3d_25d_mask_report.json"
    report_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return UNet25DRunOutput(
        sample,
        input_zarr,
        mask_zarr,
        probability_zarr,
        combined_mask_tif,
        source_slice_dir,
        preview_dir,
        report_json,
    )


__all__ = [
    "UNet25DRunConfig",
    "UNet25DRunOutput",
    "combine_source_masks_pixelwise_max",
    "combine_source_probabilities_pixelwise_max",
    "load_25d_model",
    "run_3d_25d_unet_masks",
]
