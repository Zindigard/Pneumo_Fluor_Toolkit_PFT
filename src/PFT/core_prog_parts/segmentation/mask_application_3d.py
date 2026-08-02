"""
pply a saved 2.5D U-Net mask to saved 3D deconvolution output.

This module performs the agreed final post-processing step only after both
independent branches are complete:

* original OME-Zarr -> 2.5D U-Net -> saved binary mask;
* original raw intensities -> Richardson-Lucy -> saved float32 OME-Zarr.

Foreground voxels retain 100% of the deconvolved intensity. Background voxels
retain 2% by default. ROI SNR is calculated from saved images using the same
formula as the 2D workflow and only on manually annotated target slices.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr

from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.deconvolution_no_fuji import (
    _create_output_store,
    _downsample_yx_mean,
)
from PFT.core_prog_parts.denoising.metadata_3d import copyable_root_metadata
from PFT.core_prog_parts.denoising.validation_3d import (
    annotation_sample_dir,
    find_slice_mask,
    relative_volume_path,
    target_slice_for_volume,
    volume_key,
)
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import read_binary_slice_mask


@dataclass(frozen=True)
class SNRRow:
    """ROI-SNR values for one annotated slice and fluorescence channel."""

    sample: str
    slice_1based: int
    channel_index: int
    channel_name: str
    raw_signal_mean: float
    raw_background_mean: float
    raw_background_sd: float
    raw_snr: float
    deconvolved_signal_mean: float
    deconvolved_background_mean: float
    deconvolved_background_sd: float
    deconvolved_snr: float
    masked_signal_mean: float
    masked_background_mean: float
    masked_background_sd: float
    masked_deconvolved_snr: float
    delta_deconvolution_vs_raw: float
    delta_masked_vs_deconvolved: float
    delta_masked_vs_raw: float
    signal_pixels: int
    background_pixels: int


@dataclass(frozen=True)
class MaskApplicationOutput:
    """Saved final foreground-preserving OME-Zarr and validation products."""

    output_zarr: Path
    output_dir: Path
    validation_report: Path
    snr_csv: Path | None
    snr_json: Path | None
    preview_dir: Path


def roi_snr_components(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    epsilon: float = 1e-12,
) -> tuple[int, int, float, float, float, float]:
    """Return the same ROI-SNR components used by the 2D checking workflow.

    Args:
        image (np.ndarray): Input image array to process.
        foreground_mask (np.ndarray): Array containing foreground mask.
        epsilon (float): Numerical value controlling epsilon. Defaults to ``1e-12``.

    Returns:
        tuple[int, int, float, float, float, float]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = roi_snr_components(image=image_array, foreground_mask=image_array)
    """
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(foreground_mask, dtype=bool)
    if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
        raise ValueError(f"ROI SNR expects matching 2D image/mask, got {image.shape} and {mask.shape}")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be a finite positive value, got {epsilon}")
    finite = np.isfinite(image)
    signal = image[mask & finite]
    background = image[(~mask) & finite]
    if signal.size < 1:
        raise ValueError("ROI SNR requires at least one finite foreground pixel")
    if background.size < 2:
        raise ValueError("ROI SNR requires at least two finite background pixels")
    signal_mean = float(np.mean(signal))
    background_mean = float(np.mean(background))
    background_sd = float(np.std(background, ddof=1))
    snr = float((signal_mean - background_mean) / (background_sd + epsilon))
    return int(signal.size), int(background.size), signal_mean, background_mean, background_sd, snr


def roi_snr(image: np.ndarray, foreground_mask: np.ndarray, epsilon: float = 1e-12) -> float:
    """Calculate thesis ROI SNR using the manually labelled foreground region.

    Args:
        image (np.ndarray): Input image array to process.
        foreground_mask (np.ndarray): Array containing foreground mask.
        epsilon (float): Numerical value controlling epsilon. Defaults to ``1e-12``.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = roi_snr(image=image_array, foreground_mask=image_array)
    """
    return roi_snr_components(image, foreground_mask, epsilon)[-1]


def _read_mask_zyx(mask_zarr: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read mask zyx from persistent storage.

    Args:
        mask_zarr (Path): Filesystem path used for mask Zarr.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _read_mask_zyx(mask_zarr=Path("path/to/resource"))
    """
    meta = extract_ome_zarr_meta_for_compare(mask_zarr, level=0)
    axes = str(meta.get("axes") or "").lower()
    root = zarr.open_group(str(mask_zarr), mode="r")
    array = np.asarray(root[str(meta["array_path"])])
    if axes == "czyx" and array.shape[0] == 1:
        array = array[0]
        axes = "zyx"
    if axes != "zyx" or array.ndim != 3:
        raise ValueError(f"Predicted mask must be ZYX, got axes={axes!r}, shape={array.shape}")
    unique = np.unique(array)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"Predicted mask is not binary 0/1: values={unique[:20]}")
    if unique.size < 2:
        raise ValueError(
            f"Predicted mask must contain foreground and background; found only {unique.tolist()}"
        )
    return array.astype(np.uint8), meta


def _display_composite(cyx: np.ndarray) -> np.ndarray:
    """Return display composite for the supplied inputs.

    Args:
        cyx (np.ndarray): Array containing cyx.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _display_composite(cyx=image_array)
    """
    output = np.zeros((*cyx.shape[1:], 3), dtype=np.float32)
    destinations = (2, 1, 0)
    for channel in range(min(3, cyx.shape[0])):
        low, high = np.percentile(cyx[channel], (1.0, 99.8))
        if high <= low:
            high = low + 1.0
        output[..., destinations[channel]] = np.clip((cyx[channel] - low) / (high - low), 0.0, 1.0)
    return output


def _save_previews(
    *,
    deconvolved: zarr.Array,
    masked: zarr.Array,
    mask_zyx: np.ndarray,
    sample: str,
    output_dir: Path,
    slices_1based: Sequence[int],
    manual_mask_dir: Path | None,
) -> list[Path]:
    """Save previews to persistent storage.

    Args:
        deconvolved (zarr.Array): Array containing deconvolved.
        masked (zarr.Array): Array containing masked.
        mask_zyx (np.ndarray): Array containing mask zyx.
        sample (str): Text value specifying sample.
        output_dir (Path): Directory where generated resources are written.
        slices_1based (Sequence[int]): Numerical value controlling slices 1based.
        manual_mask_dir (Path | None): Directory used for manual mask.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _save_previews(
        ...     deconvolved=image_array,
        ...     masked=image_array,
        ...     mask_zyx=image_array,
        ...     sample="sample",
        ...     output_dir=Path("path/to/resource"),
        ...     slices_1based=1,
        ...     manual_mask_dir=Path("path/to/resource"),
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for slice_number in slices_1based:
        z_index = slice_number - 1
        if not 0 <= z_index < mask_zyx.shape[0]:
            continue
        deconv_cyx = np.asarray(deconvolved[:, z_index], dtype=np.float32)
        masked_cyx = np.asarray(masked[:, z_index], dtype=np.float32)
        figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
        axes[0].imshow(_display_composite(deconv_cyx))
        axes[0].set_title("Deconvolved")
        axes[1].imshow(mask_zyx[z_index], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Saved U-Net mask")
        axes[2].imshow(_display_composite(masked_cyx))
        axes[2].set_title("Mask applied: 2% background")
        difference = masked_cyx.mean(axis=0) - deconv_cyx.mean(axis=0)
        limit = float(np.percentile(np.abs(difference), 99.5)) or 1.0
        image = axes[3].imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit)
        figure.colorbar(image, ax=axes[3], fraction=0.046)
        axes[3].set_title("Masked minus deconvolved")
        axes[4].imshow(_display_composite(deconv_cyx))
        axes[4].contour(mask_zyx[z_index], levels=[0.5], colors="white", linewidths=0.7)
        reference_path = find_slice_mask(manual_mask_dir, slice_number) if manual_mask_dir else None
        if reference_path is not None:
            reference = read_binary_slice_mask(reference_path, mask_zyx.shape[-2:])
            axes[4].contour(reference, levels=[0.5], colors="red", linewidths=0.7)
            axes[4].set_title("White: predicted | red: manual")
        else:
            axes[4].set_title("Predicted-mask overlay")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{sample} | Z{slice_number:03d}")
        safe_sample = sample.replace("/", "__").replace("\\", "__")
        path = output_dir / f"{safe_sample}__z{slice_number:03d}__mask_application_qc.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _calculate_snr_rows(
    *,
    sample: str,
    raw_zarr: Path,
    raw_level: int,
    deconvolved: zarr.Array,
    masked: zarr.Array,
    manual_mask_dir: Path,
    channel_names: Sequence[str],
    slices_1based: Sequence[int],
    epsilon: float,
) -> list[SNRRow]:
    """Calculate signal-to-noise ratio rows from the supplied data.

    Args:
        sample (str): Text value specifying sample.
        raw_zarr (Path): Filesystem path used for raw Zarr.
        raw_level (int): Numerical value controlling raw level.
        deconvolved (zarr.Array): Array containing deconvolved.
        masked (zarr.Array): Array containing masked.
        manual_mask_dir (Path): Directory used for manual mask.
        channel_names (Sequence[str]): Text value specifying channel names.
        slices_1based (Sequence[int]): Numerical value controlling slices 1based.
        epsilon (float): Numerical value controlling epsilon.

    Returns:
        list[SNRRow]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _calculate_snr_rows(
        ...     sample="sample",
        ...     raw_zarr=Path("path/to/resource"),
        ...     raw_level=1,
        ...     deconvolved=image_array,
        ...     masked=image_array,
        ...     manual_mask_dir=Path("path/to/resource"),
        ...     channel_names="channel_names",
        ...     slices_1based=1,
        ...     epsilon=0.5,
        ... )
    """
    raw_meta = extract_ome_zarr_meta_for_compare(raw_zarr, level=raw_level)
    raw_root = zarr.open_group(str(raw_zarr), mode="r")
    raw = raw_root[str(raw_meta["array_path"])]
    if tuple(raw.shape) != tuple(deconvolved.shape):
        raise ValueError(
            f"Raw/deconvolved shape mismatch for SNR: raw={raw.shape}; deconvolved={deconvolved.shape}. "
            "Use manual masks and U-Net inference at the same physical pyramid level as deconvolution."
        )
    rows: list[SNRRow] = []
    for slice_number in slices_1based:
        mask_path = find_slice_mask(manual_mask_dir, slice_number)
        if mask_path is None:
            continue
        z_index = slice_number - 1
        reference = read_binary_slice_mask(mask_path, tuple(raw.shape[-2:]))
        for channel_index in range(int(raw.shape[0])):
            raw_stats = roi_snr_components(np.asarray(raw[channel_index, z_index]), reference, epsilon)
            deconv_stats = roi_snr_components(np.asarray(deconvolved[channel_index, z_index]), reference, epsilon)
            masked_stats = roi_snr_components(np.asarray(masked[channel_index, z_index]), reference, epsilon)
            raw_value = raw_stats[-1]
            deconv_value = deconv_stats[-1]
            masked_value = masked_stats[-1]
            rows.append(SNRRow(
                sample=sample,
                slice_1based=slice_number,
                channel_index=channel_index,
                channel_name=str(channel_names[channel_index]),
                raw_signal_mean=raw_stats[2],
                raw_background_mean=raw_stats[3],
                raw_background_sd=raw_stats[4],
                raw_snr=raw_value,
                deconvolved_signal_mean=deconv_stats[2],
                deconvolved_background_mean=deconv_stats[3],
                deconvolved_background_sd=deconv_stats[4],
                deconvolved_snr=deconv_value,
                masked_signal_mean=masked_stats[2],
                masked_background_mean=masked_stats[3],
                masked_background_sd=masked_stats[4],
                masked_deconvolved_snr=masked_value,
                delta_deconvolution_vs_raw=deconv_value - raw_value,
                delta_masked_vs_deconvolved=masked_value - deconv_value,
                delta_masked_vs_raw=masked_value - raw_value,
                signal_pixels=raw_stats[0],
                background_pixels=raw_stats[1],
            ))
    return rows


def apply_saved_mask_to_deconvolution(
    *,
    deconvolved_zarr: Path,
    predicted_mask_zarr: Path,
    output_root: Path,
    manual_mask_root: Path | None = None,
    outside_mask_depletion: float = 0.98,
    pyramid_max_layer: int = 2,
    slices_1based: Sequence[int] | None = None,
    epsilon: float = 1e-12,
    overwrite: bool = True,
) -> MaskApplicationOutput:
    """Apply the saved mask, validate stored values, and calculate annotated-slice SNR.

    Args:
        deconvolved_zarr (Path): Filesystem path used for deconvolved Zarr.
        predicted_mask_zarr (Path): Filesystem path used for predicted mask Zarr.
        output_root (Path): Directory used for output.
        manual_mask_root (Path | None): Directory used for manual mask. ``None`` selects the function's default behavior.
        outside_mask_depletion (float): Numerical value controlling outside mask depletion. Defaults to ``0.98``.
        pyramid_max_layer (int): Numerical value controlling pyramid max layer. Defaults to ``2``.
        slices_1based (Sequence[int] | None): Numerical value controlling slices 1based. ``None`` selects the function's default behavior.
        epsilon (float): Numerical value controlling epsilon. Defaults to ``1e-12``.
        overwrite (bool): Whether an existing output may be replaced. Defaults to ``True``.

    Returns:
        MaskApplicationOutput: Result produced by the operation.

    Raises:
        FileExistsError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = apply_saved_mask_to_deconvolution(
        ...     deconvolved_zarr=Path("path/to/resource"),
        ...     predicted_mask_zarr=Path("path/to/resource"),
        ...     output_root=Path("path/to/resource"),
        ... )
    """
    if not 0.0 <= outside_mask_depletion <= 1.0:
        raise ValueError("outside_mask_depletion must be in [0,1]")
    residual = 1.0 - float(outside_mask_depletion)
    deconvolved_zarr = Path(deconvolved_zarr).resolve()
    predicted_mask_zarr = Path(predicted_mask_zarr).resolve()
    output_root = Path(output_root).resolve()

    deconv_meta = extract_ome_zarr_meta_for_compare(deconvolved_zarr, level=0)
    if str(deconv_meta.get("axes") or "").lower() != "czyx":
        raise ValueError(f"Deconvolved image must be CZYX: {deconv_meta.get('axes')}")
    deconv_root = zarr.open_group(str(deconvolved_zarr), mode="r")
    deconv = deconv_root[str(deconv_meta["array_path"])]
    mask, _ = _read_mask_zyx(predicted_mask_zarr)
    if tuple(mask.shape) != tuple(deconv.shape[1:]):
        raise ValueError(
            f"Predicted mask and deconvolution must have identical ZYX geometry: "
            f"mask={mask.shape}; deconvolved={deconv.shape[1:]}."
        )

    processing_source = deconv_root.attrs.get("pft_processing")
    source_raw = None
    source_level = 0
    if isinstance(processing_source, dict):
        source_raw = processing_source.get("source_omezarr")
        source_level = int(processing_source.get("source_level", 0))

    mask_root_group = zarr.open_group(str(predicted_mask_zarr), mode="r")
    mask_processing = mask_root_group.attrs.get("pft_processing")
    mask_source = mask_processing.get("source_omezarr") if isinstance(mask_processing, dict) else None
    mask_level = int(mask_processing.get("source_level", 0)) if isinstance(mask_processing, dict) else None
    if source_raw is not None and mask_source is not None:
        if Path(source_raw).resolve() != Path(mask_source).resolve():
            raise ValueError(
                "Predicted mask and deconvolution were derived from different source images: "
                f"deconvolution={source_raw}; mask={mask_source}"
            )
        if mask_level != source_level:
            raise ValueError(
                "Predicted mask and deconvolution used different source pyramid levels: "
                f"deconvolution=L{source_level}; mask=L{mask_level}"
            )
    elif source_raw is None or mask_source is None:
        raise ValueError(
            "Source provenance is missing from the deconvolution or predicted-mask metadata; "
            "refusing to combine arrays using shape alone."
        )

    source_raw_path = Path(source_raw).resolve()
    relative_volume = relative_volume_path(source_raw_path)
    sample = volume_key(source_raw_path)
    if slices_1based is None:
        slices_1based = (target_slice_for_volume(source_raw_path),)
    slices_1based = tuple(sorted({int(value) for value in slices_1based}))
    if not slices_1based:
        raise ValueError("At least one target slice is required for SNR calculation")
    output_dir = output_root / relative_volume
    output_zarr = output_dir / "image.ome.zarr"
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retained = copyable_root_metadata(deconvolved_zarr)
    multiscales = deconv_root.attrs.get("multiscales")
    base_scale = [1.0, 1.0, 1.0, 1.0]
    if isinstance(multiscales, list) and multiscales:
        transformations = multiscales[0]["datasets"][0].get("coordinateTransformations", [])
        for transformation in transformations:
            if transformation.get("type") == "scale":
                base_scale = [float(value) for value in transformation["scale"]]
                break
    processing = {
        "operation": "apply_unet_3d_25d_foreground_mask",
        "source_deconvolved_omezarr": str(deconvolved_zarr),
        "source_predicted_mask_omezarr": str(predicted_mask_zarr),
        "outside_mask_depletion_fraction": float(outside_mask_depletion),
        "outside_mask_residual_fraction": residual,
        "inside_mask_intensity_scale": 1.0,
        "normalization_applied_to_stored_intensities": False,
        "stored_dtype": "float32",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    chunks = tuple(int(value) for value in (deconv.chunks or (1, 1, 256, 256)))
    _, output_arrays = _create_output_store(
        out_zarr=output_zarr,
        shape_czyx=tuple(int(value) for value in deconv.shape),
        chunks_czyx=chunks,
        base_scale=base_scale,
        pyramid_max_layer=pyramid_max_layer,
        source_attrs=retained,
        processing=processing,
    )

    validation: list[dict[str, Any]] = []
    mask_bool = mask.astype(bool)
    for channel_index in range(int(deconv.shape[0])):
        source = np.asarray(deconv[channel_index], dtype=np.float32)
        result = np.where(mask_bool, source, source * np.float32(residual)).astype(np.float32)
        output_arrays[0][channel_index] = result
        level_data = result
        for level in range(1, pyramid_max_layer + 1):
            level_data = _downsample_yx_mean(level_data, factor=2)
            expected = output_arrays[level].shape[1:]
            if level_data.shape != expected:
                level_data = level_data[:, : expected[1], : expected[2]]
            output_arrays[level][channel_index] = level_data
        stored = np.asarray(output_arrays[0][channel_index])
        foreground_ok = bool(np.array_equal(stored[mask_bool], source[mask_bool]))
        background_ok = bool(np.allclose(stored[~mask_bool], source[~mask_bool] * residual, rtol=1e-6, atol=1e-7))
        validation.append({
            "channel_index": channel_index,
            "foreground_values_unchanged": foreground_ok,
            "background_values_match_residual": background_ok,
            "stored_dtype": str(stored.dtype),
            "stored_finite": bool(np.isfinite(stored).all()),
            "stored_min": float(stored.min()),
            "stored_max": float(stored.max()),
        })

    stored_root = zarr.open_group(str(output_zarr), mode="r")
    stored_level0 = stored_root["0"]
    preview_dir = output_dir / "qc_training_slices"
    previews = _save_previews(
        deconvolved=deconv,
        masked=stored_level0,
        mask_zyx=mask,
        sample=sample,
        output_dir=preview_dir,
        slices_1based=slices_1based,
        manual_mask_dir=(
            annotation_sample_dir(Path(manual_mask_root), source_raw_path)
            if manual_mask_root
            else None
        ),
    )

    snr_csv: Path | None = None
    snr_json: Path | None = None
    snr_rows: list[SNRRow] = []
    channel_names = deconv_root.attrs.get("channel_names") or [f"channel_{index}" for index in range(int(deconv.shape[0]))]
    if manual_mask_root is not None and source_raw is not None:
        snr_rows = _calculate_snr_rows(
            sample=sample,
            raw_zarr=source_raw_path,
            raw_level=source_level,
            deconvolved=deconv,
            masked=stored_level0,
            manual_mask_dir=annotation_sample_dir(Path(manual_mask_root), source_raw_path),
            channel_names=channel_names,
            slices_1based=slices_1based,
            epsilon=epsilon,
        )
        if snr_rows:
            snr_csv = output_dir / "snr_training_slices.csv"
            with snr_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(snr_rows[0]).keys()))
                writer.writeheader()
                writer.writerows(asdict(row) for row in snr_rows)
            snr_summary = {
                "formula": "(mean foreground - mean background) / (sample SD background + epsilon)",
                "epsilon": epsilon,
                "slices_1based": list(slices_1based),
                "rows": [asdict(row) for row in snr_rows],
                "mean_raw_snr": float(np.mean([row.raw_snr for row in snr_rows])),
                "mean_deconvolved_snr": float(np.mean([row.deconvolved_snr for row in snr_rows])),
                "mean_masked_deconvolved_snr": float(np.mean([row.masked_deconvolved_snr for row in snr_rows])),
            }
            snr_json = output_dir / "snr_training_slices.json"
            snr_json.write_text(json.dumps(snr_summary, indent=2), encoding="utf-8")

    passed = all(
        item["foreground_values_unchanged"]
        and item["background_values_match_residual"]
        and item["stored_finite"]
        and item["stored_dtype"] == "float32"
        for item in validation
    )
    validation_report = output_dir / "mask_application_validation.txt"
    lines = [
        "PFT 3D U-Net mask application validation",
        "=" * 80,
        f"Status: {'PASS' if passed else 'FAIL'}",
        f"Deconvolved input: {deconvolved_zarr}",
        f"Predicted mask: {predicted_mask_zarr}",
        f"Output OME-Zarr: {output_zarr}",
        f"Foreground scale: 1.0",
        f"Background residual: {residual}",
        "Stored normalization: NONE",
        "",
    ]
    for item in validation:
        lines.append(json.dumps(item, sort_keys=True))
    lines.extend(["", f"SNR rows: {len(snr_rows)}", f"QC previews: {len(previews)}"])
    validation_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError(f"Mask application failed validation; see {validation_report}")
    return MaskApplicationOutput(output_zarr, output_dir, validation_report, snr_csv, snr_json, preview_dir)


__all__ = [
    "MaskApplicationOutput",
    "SNRRow",
    "apply_saved_mask_to_deconvolution",
    "roi_snr",
    "roi_snr_components",
]
