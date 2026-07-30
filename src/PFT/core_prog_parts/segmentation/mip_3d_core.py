"""
Create 2D maximum-intensity projections from PFT 3D OME-Zarr volumes.

All 40 Z planes are projected. The saved quantitative image remains in the
source intensity scale: raw products retain the original dtype, whereas the
selected deconvolution product is stored as float32. Display normalization is
used only for the merged-RGB QC figure.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.metadata_3d import (
    coordinate_scale_for_level,
    copyable_root_metadata,
    resolve_channel_optics,
)
from PFT.core_prog_parts.denoising.validation_3d import (
    relative_volume_path,
    target_slice_for_volume,
    volume_key,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
MIPMode = Literal["raw_unmasked", "raw_masked", "deconv_masked"]

DECONV_MODEL = "BW"
DECONV_ITERATION_TAG = "iterB3_G3_R2"
DECONV_SOURCE_LEVEL = 0
EXPECTED_Z_COUNT = 40

MODE_DIRECTORY_NAMES: dict[MIPMode, str] = {
    "raw_unmasked": "raw_mip",
    "raw_masked": "raw_mip_masked",
    "deconv_masked": "deconv_332_mip_masked",
}


@dataclass
class MIP3DConfig:
    """Configuration for one raw source volume and one requested MIP product."""

    project_root: Path = find_project_root(Path(__file__).resolve())
    raw_zarr: Path | None = None
    mode: MIPMode = "raw_unmasked"
    raw_root: Path | None = None
    deconv_root: Path | None = None
    mask_root: Path | None = None
    output_root: Path | None = None
    level: int = 0
    expected_z_count: int = EXPECTED_Z_COUNT
    overwrite: bool = True
    qc_percentile_low: float = 1.0
    qc_percentile_high: float = 99.8
    qc_max_dimension: int = 1400


@dataclass(frozen=True)
class MIP3DOutput:
    """Saved files and essential provenance for one MIP product."""

    sample: str
    mode: str
    raw_zarr: Path
    projection_source_zarr: Path
    mask_zarr: Path | None
    output_zarr: Path
    qc_png: Path
    report_json: Path
    target_slice_1based: int
    output_dtype: str
    output_shape_cyx: tuple[int, int, int]



RGB_REFERENCE_WAVELENGTHS_NM: dict[str, float] = {
    "red": 561.0,
    "green": 488.0,
    "blue": 405.0,
}


def resolve_rgb_source_channels(image_zarr: Path, *, level: int = 0) -> tuple[int, int, int]:
    """Resolve source channels in red, green, blue order from wavelength metadata."""
    assignments: dict[str, int] = {}
    for item in resolve_channel_optics(image_zarr, level=level):
        colour, reference_nm = min(
            RGB_REFERENCE_WAVELENGTHS_NM.items(),
            key=lambda pair: abs(float(item.wavelength_nm) - pair[1]),
        )
        if abs(float(item.wavelength_nm) - reference_nm) > 35.0:
            raise ValueError(
                f"Channel {item.index} ({item.name}) at {item.wavelength_nm:g} nm "
                "cannot be mapped safely to 405/488/561 nm RGB."
            )
        if colour in assignments:
            raise ValueError(f"Multiple source channels map to {colour}")
        assignments[colour] = int(item.index)
    missing = [colour for colour in ("red", "green", "blue") if colour not in assignments]
    if missing:
        raise ValueError(f"Missing wavelength channel(s) for merged RGB display: {missing}")
    return assignments["red"], assignments["green"], assignments["blue"]


def _require_mode(mode: str) -> MIPMode:
    if mode not in MODE_DIRECTORY_NAMES:
        raise ValueError(
            f"Unknown MIP mode {mode!r}; expected one of {tuple(MODE_DIRECTORY_NAMES)}"
        )
    return mode  # type: ignore[return-value]


def _open_level_array(zarr_path: Path, level: int) -> tuple[Any, str, dict[str, Any]]:
    """Open one OME-Zarr level lazily and return its array, axes, and metadata."""
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "MIP creation requires zarr and ome-zarr from the project environment."
        ) from exc

    zarr_path = Path(zarr_path).expanduser().resolve()
    if not zarr_path.is_dir() or not zarr_path.name.endswith(".ome.zarr"):
        raise FileNotFoundError(f"Expected an OME-Zarr directory: {zarr_path}")

    metadata = extract_ome_zarr_meta_for_compare(zarr_path, level=level)
    axes = str(metadata.get("axes") or "").lower()
    array_path = str(metadata.get("array_path") or level)
    root = zarr.open_group(str(zarr_path), mode="r")
    array = root[array_path]
    if len(axes) != array.ndim:
        raise ValueError(
            f"OME axes {axes!r} do not match shape {tuple(array.shape)} for {zarr_path}"
        )
    return array, axes, metadata


def _require_czyx(array: Any, axes: str, source: Path, expected_z_count: int) -> None:
    """Validate the thesis 3D source representation without loading the volume."""
    if axes != "czyx":
        raise ValueError(f"Expected CZYX OME-Zarr, received axes={axes!r}: {source}")
    if int(array.shape[0]) != 3:
        raise ValueError(f"Expected three fluorescence channels, got shape={array.shape}: {source}")
    if int(array.shape[1]) != int(expected_z_count):
        raise ValueError(
            f"Expected exactly {expected_z_count} Z planes, got Z={array.shape[1]}: {source}"
        )
    if int(array.shape[2]) < 1 or int(array.shape[3]) < 1:
        raise ValueError(f"Invalid spatial shape {array.shape}: {source}")


def maximum_intensity_projection_cyx(array_czyx: Any) -> np.ndarray:
    """Calculate a C×Y×X maximum projection while loading only one Z plane at a time."""
    if getattr(array_czyx, "ndim", None) != 4:
        raise ValueError(f"Expected a four-dimensional CZYX array, got {array_czyx.shape}")
    c_count, z_count, y_count, x_count = (int(value) for value in array_czyx.shape)
    if z_count < 1:
        raise ValueError("Cannot create a MIP from an empty Z axis")

    output = np.empty((c_count, y_count, x_count), dtype=np.dtype(array_czyx.dtype))
    for channel in range(c_count):
        channel_mip = np.asarray(array_czyx[channel, 0, :, :]).copy()
        if not np.isfinite(channel_mip).all():
            raise ValueError(f"Non-finite values in channel {channel}, Z1")
        for z_index in range(1, z_count):
            plane = np.asarray(array_czyx[channel, z_index, :, :])
            if not np.isfinite(plane).all():
                raise ValueError(
                    f"Non-finite values in channel {channel}, Z{z_index + 1}"
                )
            np.maximum(channel_mip, plane, out=channel_mip)
        output[channel] = channel_mip
    return output


def _read_cyx_slice(array_czyx: Any, slice_1based: int) -> np.ndarray:
    """Read one target slice from a CZYX array without changing its stored scale."""
    z_index = int(slice_1based) - 1
    if not 0 <= z_index < int(array_czyx.shape[1]):
        raise IndexError(
            f"Target Z{slice_1based} is outside a stack with Z={array_czyx.shape[1]}"
        )
    plane = np.asarray(array_czyx[:, z_index, :, :])
    if not np.isfinite(plane).all():
        raise ValueError(f"Non-finite values in configured raw target Z{slice_1based}")
    return plane


def resolve_deconvolved_332_zarr(raw_zarr: Path, deconv_root: Path) -> Path:
    """Resolve the exact BW Richardson-Lucy 3/3/2 result for one raw volume."""
    relative = relative_volume_path(raw_zarr)
    sample = relative.name
    result = (
        Path(deconv_root)
        / relative.parent
        / (
            f"{sample}__SK_RL__PSF{DECONV_MODEL}__{DECONV_ITERATION_TAG}"
            f"__sourceL{DECONV_SOURCE_LEVEL}"
        )
        / "image.ome.zarr"
    )
    if not result.is_dir():
        raise FileNotFoundError(
            "Required deconvolution result is missing. Expected the fixed BW 3/3/2 output: "
            f"{result}"
        )
    return result.resolve()


def resolve_predicted_mask_zarr(raw_zarr: Path, mask_root: Path) -> Path:
    """Resolve the existing per-volume 2.5D U-Net mask for one raw source."""
    result = Path(mask_root) / relative_volume_path(raw_zarr) / "pred_mask.ome.zarr"
    if not result.is_dir():
        raise FileNotFoundError(
            "The existing 2.5D U-Net prediction is missing. Run 2.5D inference first: "
            f"{result}"
        )
    return result.resolve()


def _load_target_mask_yx(
    mask_zarr: Path,
    *,
    target_slice_1based: int,
    expected_yx: tuple[int, int],
) -> np.ndarray:
    """Read the target plane from a YX or broadcast ZYX binary mask OME-Zarr."""
    array, axes, _metadata = _open_level_array(mask_zarr, level=0)
    if axes == "yx":
        mask = np.asarray(array)
    elif axes == "zyx":
        z_index = int(target_slice_1based) - 1
        if not 0 <= z_index < int(array.shape[0]):
            raise IndexError(
                f"Mask does not contain target Z{target_slice_1based}: shape={array.shape}"
            )
        mask = np.asarray(array[z_index, :, :])
    else:
        raise ValueError(f"Expected YX or ZYX predicted mask, received axes={axes!r}: {mask_zarr}")

    if tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(
            f"Predicted-mask/image mismatch: mask={mask.shape}, expected={expected_yx}: {mask_zarr}"
        )
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary) or np.all(binary):
        raise ValueError(f"Predicted mask must contain foreground and background: {mask_zarr}")
    return binary


def apply_zero_background_mask(mip_cyx: np.ndarray, mask_yx: np.ndarray) -> np.ndarray:
    """Set every channel outside the 2D mask to exactly zero and preserve dtype."""
    if mip_cyx.ndim != 3:
        raise ValueError(f"Expected CYX MIP, got shape={mip_cyx.shape}")
    if tuple(mip_cyx.shape[1:]) != tuple(mask_yx.shape):
        raise ValueError(f"MIP/mask shape mismatch: {mip_cyx.shape} vs {mask_yx.shape}")
    return np.where(mask_yx[None, :, :] > 0, mip_cyx, 0).astype(
        mip_cyx.dtype, copy=False
    )


def _cyx_coordinate_scale(source_zarr: Path, *, level: int) -> list[float]:
    """Drop the projected Z scale while preserving C, Y, and X sampling."""
    metadata = extract_ome_zarr_meta_for_compare(source_zarr, level=level)
    axes = str(metadata.get("axes") or "").lower()
    scale = coordinate_scale_for_level(source_zarr, level=level)
    if len(scale) != len(axes):
        raise ValueError(f"Coordinate scale {scale} does not match axes {axes!r}")
    if not all(axis in axes for axis in "cyx"):
        raise ValueError(f"Cannot derive CYX scale from axes={axes!r}")
    return [float(scale[axes.index(axis)]) for axis in "cyx"]


def _to_rgb_hwc(cyx: np.ndarray, rgb_source_channels: Sequence[int]) -> np.ndarray:
    """Map source channels to red/green/blue display order without normalization."""
    if len(rgb_source_channels) != 3:
        raise ValueError(f"Expected three RGB source indices, got {rgb_source_channels}")
    return np.stack(
        [np.asarray(cyx[int(channel)], dtype=np.float32) for channel in rgb_source_channels],
        axis=-1,
    )


def _raw_scale_rgb(
    cyx_images: Sequence[np.ndarray],
    rgb_source_channels: Sequence[int],
    percentile_high: float,
) -> list[np.ndarray]:
    """Create merged RGB views using one shared zero-based scale per colour."""
    rgb_images = [_to_rgb_hwc(image, rgb_source_channels) for image in cyx_images]
    high_limits: list[float] = []
    for rgb_channel in range(3):
        candidate_limits = [
            float(np.percentile(image[..., rgb_channel], percentile_high))
            for image in rgb_images
        ]
        high = max(candidate_limits)
        high_limits.append(high if np.isfinite(high) and high > 0 else 1.0)

    outputs: list[np.ndarray] = []
    for image in rgb_images:
        normalized = np.empty_like(image, dtype=np.float32)
        for rgb_channel, high in enumerate(high_limits):
            normalized[..., rgb_channel] = np.clip(
                image[..., rgb_channel] / high,
                0.0,
                1.0,
            )
        outputs.append(normalized)
    return outputs


def _independently_normalized_rgb(
    cyx: np.ndarray,
    rgb_source_channels: Sequence[int],
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    """Percentile-normalize each colour independently for display only."""
    rgb = _to_rgb_hwc(cyx, rgb_source_channels)
    output = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[..., channel]
        low = float(np.percentile(plane, percentile_low))
        high = float(np.percentile(plane, percentile_high))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            output[..., channel] = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return output


def _downsample_cyx_for_qc(cyx: np.ndarray, max_dimension: int) -> np.ndarray:
    """Reduce display-only arrays by integer stride while retaining the full saved MIP."""
    if max_dimension < 256:
        raise ValueError("qc_max_dimension must be at least 256 pixels")
    largest = max(int(cyx.shape[-2]), int(cyx.shape[-1]))
    stride = max(1, int(np.ceil(largest / float(max_dimension))))
    return cyx[:, ::stride, ::stride]


def _save_merged_qc(
    *,
    raw_target_cyx: np.ndarray,
    mip_before_cyx: np.ndarray,
    mip_output_cyx: np.ndarray,
    rgb_source_channels: Sequence[int],
    target_slice_1based: int,
    sample: str,
    mode: MIPMode,
    output_png: Path,
    percentile_low: float,
    percentile_high: float,
    max_dimension: int,
) -> Path:
    """Save merged-only raw-scale and normalized MIP comparisons."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    display_images = [
        _downsample_cyx_for_qc(image, max_dimension)
        for image in (raw_target_cyx, mip_before_cyx, mip_output_cyx)
    ]
    raw_scale = _raw_scale_rgb(
        display_images,
        rgb_source_channels,
        percentile_high,
    )
    normalized = [
        _independently_normalized_rgb(
            image,
            rgb_source_channels,
            percentile_low,
            percentile_high,
        )
        for image in display_images
    ]

    output_title = "Masked MIP (outside=0)" if mode != "raw_unmasked" else "Output MIP (unmasked)"
    column_titles = (
        f"Raw target Z{target_slice_1based}",
        "MIP before mask",
        output_title,
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for column, title in enumerate(column_titles):
        axes[0, column].imshow(raw_scale[column])
        axes[0, column].set_title(f"{title}\nshared raw-intensity display scale")
        axes[1, column].imshow(normalized[column])
        axes[1, column].set_title(f"{title}\nindependent P{percentile_low:g}–P{percentile_high:g}")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{sample} | {mode} | merged RGB only")
    figure.savefig(output_png, dpi=170)
    plt.close(figure)
    return output_png


def _output_sample_dir(cfg: MIP3DConfig, raw_zarr: Path) -> Path:
    output_root = Path(cfg.output_root or cfg.project_root / "results" / "mip_2d")
    return output_root / MODE_DIRECTORY_NAMES[_require_mode(cfg.mode)] / relative_volume_path(raw_zarr)


def create_mip_for_volume(cfg: MIP3DConfig) -> MIP3DOutput:
    """Create and save one selected 2D MIP product for one configured raw volume."""
    if cfg.raw_zarr is None:
        raise ValueError("raw_zarr is required")
    if int(cfg.level) != 0:
        raise ValueError("The current 3D MIP workflow is fixed to source level 0")
    mode = _require_mode(cfg.mode)

    raw_zarr = Path(cfg.raw_zarr).expanduser().resolve()
    raw_root = Path(cfg.raw_root or cfg.project_root / "results" / "img" / "3d_data")
    deconv_root = Path(cfg.deconv_root or cfg.project_root / "results" / "deconv")
    mask_root = Path(cfg.mask_root or cfg.project_root / "results" / "U-net" / "3d_25d")

    # Validate that the raw source belongs to the configured mapping and read the
    # raw target before selecting a possibly deconvolved projection source.
    target_slice = target_slice_for_volume(raw_zarr, image_root=raw_root)
    sample = volume_key(raw_zarr, image_root=raw_root)
    raw_array, raw_axes, _raw_metadata = _open_level_array(raw_zarr, cfg.level)
    _require_czyx(raw_array, raw_axes, raw_zarr, cfg.expected_z_count)
    raw_target_cyx = _read_cyx_slice(raw_array, target_slice)

    projection_source_zarr = (
        resolve_deconvolved_332_zarr(raw_zarr, deconv_root)
        if mode == "deconv_masked"
        else raw_zarr
    )
    source_array, source_axes, _source_metadata = _open_level_array(
        projection_source_zarr, cfg.level
    )
    _require_czyx(
        source_array,
        source_axes,
        projection_source_zarr,
        cfg.expected_z_count,
    )
    mip_before = maximum_intensity_projection_cyx(source_array)

    if mode == "deconv_masked":
        mip_before = mip_before.astype(np.float32, copy=False)
        if mip_before.dtype != np.float32:
            raise ValueError("Deconvolved MIP must be float32")

    mask_zarr: Path | None = None
    mask_yx: np.ndarray | None = None
    if mode in ("raw_masked", "deconv_masked"):
        mask_zarr = resolve_predicted_mask_zarr(raw_zarr, mask_root)
        mask_yx = _load_target_mask_yx(
            mask_zarr,
            target_slice_1based=target_slice,
            expected_yx=tuple(int(value) for value in mip_before.shape[1:]),
        )
        mip_output = apply_zero_background_mask(mip_before, mask_yx)
    else:
        mip_output = mip_before.copy()

    if mode.startswith("raw_") and mip_output.dtype != np.dtype(raw_array.dtype):
        raise ValueError(
            f"Raw MIP dtype changed unexpectedly: {mip_output.dtype} != {raw_array.dtype}"
        )
    if mode == "deconv_masked" and mip_output.dtype != np.float32:
        raise ValueError(f"Deconvolved MIP must be float32, got {mip_output.dtype}")

    sample_dir = _output_sample_dir(cfg, raw_zarr)
    if sample_dir.exists() and cfg.overwrite:
        shutil.rmtree(sample_dir)
    elif sample_dir.exists():
        raise FileExistsError(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    retained = copyable_root_metadata(projection_source_zarr)
    processing = {
        "operation": "maximum_intensity_projection_3d_to_2d",
        "mode": mode,
        "raw_source_omezarr": str(raw_zarr),
        "projection_source_omezarr": str(projection_source_zarr),
        "source_level": int(cfg.level),
        "projected_axis": "z",
        "projected_z_count": int(source_array.shape[1]),
        "projected_z_range_1based": [1, int(source_array.shape[1])],
        "target_slice_1based_for_qc_and_mask": int(target_slice),
        "mask_omezarr": str(mask_zarr) if mask_zarr is not None else None,
        "mask_source": "existing_2.5d_unet_target_prediction" if mask_zarr else None,
        "mask_applied_after_projection": bool(mask_zarr),
        "outside_mask_value": 0 if mask_zarr else None,
        "deconvolution_model": DECONV_MODEL if mode == "deconv_masked" else None,
        "deconvolution_iterations": (
            {"blue": 3, "green": 3, "red": 2}
            if mode == "deconv_masked"
            else None
        ),
        "stored_normalization": "none",
        "stored_dtype_policy": (
            "float32_from_deconvolution"
            if mode == "deconv_masked"
            else "preserve_raw_source_dtype"
        ),
        "display_qc": "merged_rgb_only; shared raw scale and independent percentile normalization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_attrs = dict(retained)
    output_attrs["pft_processing"] = processing
    output_zarr = save_ome_zarr(
        sample_dir / "image.ome.zarr",
        mip_output,
        "cyx",
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        coordinate_scale=_cyx_coordinate_scale(projection_source_zarr, level=cfg.level),
        extra_attrs=output_attrs,
    )

    rgb_source_channels = resolve_rgb_source_channels(raw_zarr, level=cfg.level)
    qc_png = _save_merged_qc(
        raw_target_cyx=raw_target_cyx,
        mip_before_cyx=mip_before,
        mip_output_cyx=mip_output,
        rgb_source_channels=rgb_source_channels,
        target_slice_1based=target_slice,
        sample=sample,
        mode=mode,
        output_png=sample_dir / "qc" / "merged_mip_vs_raw_target.png",
        percentile_low=cfg.qc_percentile_low,
        percentile_high=cfg.qc_percentile_high,
        max_dimension=cfg.qc_max_dimension,
    )

    report = {
        "sample": sample,
        "mode": mode,
        "raw_zarr": str(raw_zarr),
        "projection_source_zarr": str(projection_source_zarr),
        "mask_zarr": str(mask_zarr) if mask_zarr else None,
        "output_zarr": str(output_zarr),
        "qc_png": str(qc_png),
        "target_slice_1based": int(target_slice),
        "source_shape_czyx": [int(value) for value in source_array.shape],
        "source_dtype": str(np.dtype(source_array.dtype)),
        "mip_before_shape_cyx": list(mip_before.shape),
        "mip_before_dtype": str(mip_before.dtype),
        "output_shape_cyx": list(mip_output.shape),
        "output_dtype": str(mip_output.dtype),
        "output_min": float(np.min(mip_output)),
        "output_max": float(np.max(mip_output)),
        "foreground_pixels": int(np.count_nonzero(mask_yx)) if mask_yx is not None else None,
        "background_pixels": int(mask_yx.size - np.count_nonzero(mask_yx)) if mask_yx is not None else None,
        "configuration": asdict(cfg),
        "processing": processing,
    }
    report_json = sample_dir / "mip_3d_to_2d_report.json"
    report_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return MIP3DOutput(
        sample=sample,
        mode=mode,
        raw_zarr=raw_zarr,
        projection_source_zarr=projection_source_zarr,
        mask_zarr=mask_zarr,
        output_zarr=output_zarr,
        qc_png=qc_png,
        report_json=report_json,
        target_slice_1based=int(target_slice),
        output_dtype=str(mip_output.dtype),
        output_shape_cyx=tuple(int(value) for value in mip_output.shape),
    )


__all__ = [
    "DECONV_ITERATION_TAG",
    "DECONV_MODEL",
    "EXPECTED_Z_COUNT",
    "MIP3DConfig",
    "MIP3DOutput",
    "MIPMode",
    "MODE_DIRECTORY_NAMES",
    "apply_zero_background_mask",
    "create_mip_for_volume",
    "maximum_intensity_projection_cyx",
    "resolve_deconvolved_332_zarr",
    "resolve_predicted_mask_zarr",
]
