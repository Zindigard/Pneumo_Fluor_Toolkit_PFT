"""
Metadata-preserving 3D Richardson-Lucy deconvolution for OME-Zarr.

The selected input pyramid level is deconvolved channel by channel and written
as level 0 of a new multiscale OME-Zarr. Numeric fluorescence values are saved
as raw ``float32`` Richardson-Lucy output. No 0-1 normalization is applied to
the stored arrays. Display normalization is used only in PNG quality-control
figures. Each wavelength-matched channel may use its own iteration count.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import zarr
from skimage.restoration import richardson_lucy

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.metadata_3d import (
    coordinate_scale_for_level,
    copyable_root_metadata,
    resolve_channel_optics,
)
from PFT.core_prog_parts.denoising.validation_3d import (
    check_3d_sample,
    relative_volume_path,
    target_slice_for_volume,
    write_readiness_report,
)
from PFT.core_prog_parts.denoising.psf_creator import (
    MASTER_LEVEL,
    MasterPSFMatch,
    validate_master_psf_file,
    validate_stack_against_master_psfs,
)

PSFModel = Literal["BW", "GL", "RW"]

# These attributes describe the current derived OME-Zarr and therefore cannot
# retain the source values under the same keys when dtype, shape, or pyramid
# properties change. Their original values are preserved in a nested snapshot.
_OUTPUT_OWNED_ROOT_ATTRS: frozenset[str] = frozenset(
    {
        "pft_axes",
        "pft_level0_shape",
        "pft_level0_dtype",
        "pft_multiscale_enabled",
        "pft_pyramid_max_layer",
        "pft_pyramid_downscale",
    }
)
_SOURCE_DERIVED_METADATA_KEY = "pft_source_derived_root_metadata"


@dataclass(frozen=True)
class VolumeStats:
    """Descriptive statistics for one channel volume."""

    stage: str
    channel_index: int
    channel_name: str
    dtype: str
    voxel_count: int
    finite_fraction: float
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    sum: float
    negative_voxels: int
    zero_voxels: int


@dataclass(frozen=True)
class SkimageDeconvRunInfo:
    """Paths and parameters of one completed deconvolution run."""

    in_zarr: Path
    out_zarr: Path
    out_dir: Path
    model: PSFModel
    iters: int
    iterations_by_channel: dict[str, int]
    background: float
    input_level: int
    pyramid_max_layer: int
    out_dtype: str
    clip: bool
    filter_epsilon: float | None
    report_txt: Path
    report_json: Path
    stats_csv: Path
    preview_dir: Path
    preflight_report: Path


def _dataset_path_for_level(zarr_dir: Path, level: int) -> str:
    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    return str(meta["array_path"])


def _prepare_image_for_rl(image_zyx: np.ndarray, background: float) -> np.ndarray:
    """Convert to finite non-negative float32 values without normalization."""
    image = np.asarray(image_zyx, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Richardson-Lucy input must be ZYX, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Input volume contains NaN or infinite values")
    image = image - np.float32(background)
    return np.maximum(image, np.float32(0.0))


def _load_normalized_psf(match: MasterPSFMatch) -> np.ndarray:
    """Load one validated reusable master PSF for a wavelength-matched channel."""
    validation = validate_master_psf_file(match.psf_path, normalize_if_needed=True)
    if not validation.suitable:
        raise ValueError(
            f"Master PSF is missing or invalid: {match.psf_path}\n"
            + "\n".join(f"- {issue}" for issue in validation.issues)
        )
    psf = np.asarray(tiff.imread(str(match.psf_path)), dtype=np.float32)
    psf /= np.float32(psf.sum(dtype=np.float64))
    return psf


def _resolve_channel_iterations(
    matches: Sequence[MasterPSFMatch],
    *,
    default_iters: int,
    overrides: Mapping[str, int] | None,
) -> dict[int, int]:
    """Resolve a positive Richardson-Lucy iteration count for every channel.

    Overrides may be keyed by display color (``blue``, ``green``, ``red``),
    wavelength (for example ``405`` or ``405nm``), channel name, channel index
    (``0``), or indexed label (``c0``). Color keys are recommended because the
    master-PSF metadata already defines the wavelength-to-color convention.
    """
    if int(default_iters) < 1:
        raise ValueError("default_iters must be at least 1")

    normalized: dict[str, int] = {}
    for raw_key, raw_value in (overrides or {}).items():
        key = str(raw_key).strip().lower().replace(" ", "")
        value = int(raw_value)
        if not key:
            raise ValueError("Channel-iteration override contains an empty key")
        if value < 1:
            raise ValueError(f"Iteration count for {raw_key!r} must be at least 1")
        normalized[key] = value

    resolved: dict[int, int] = {}
    used_keys: set[str] = set()
    for match in matches:
        wavelength = float(match.wavelength_nm)
        wavelength_int = int(round(wavelength))
        candidates = (
            str(match.display_color).strip().lower(),
            str(wavelength_int),
            f"{wavelength_int}nm",
            str(match.channel_name).strip().lower().replace(" ", ""),
            str(int(match.channel_index)),
            f"c{int(match.channel_index)}",
        )
        selected = int(default_iters)
        for candidate in candidates:
            if candidate in normalized:
                selected = normalized[candidate]
                used_keys.add(candidate)
                break
        resolved[int(match.channel_index)] = selected

    unused = sorted(set(normalized) - used_keys)
    if unused:
        raise ValueError(
            "Unknown channel-iteration override key(s): " + ", ".join(unused)
        )
    return resolved


def _iteration_tag(
    matches: Sequence[MasterPSFMatch],
    iterations_by_index: Mapping[int, int],
) -> str:
    """Return a stable output-name tag such as ``iterB4_G5_R3``."""
    by_color = {
        str(match.display_color).lower(): int(iterations_by_index[match.channel_index])
        for match in matches
    }
    labels = (("blue", "B"), ("green", "G"), ("red", "R"))
    return "iter" + "_".join(
        f"{short}{by_color[color]}" for color, short in labels if color in by_color
    )


def _volume_stats(stage: str, channel_index: int, channel_name: str, array: np.ndarray) -> VolumeStats:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError(f"{stage} channel {channel_name} contains no finite voxels")
    return VolumeStats(
        stage=stage,
        channel_index=channel_index,
        channel_name=channel_name,
        dtype=str(values.dtype),
        voxel_count=int(values.size),
        finite_fraction=float(finite.mean()),
        minimum=float(finite_values.min()),
        maximum=float(finite_values.max()),
        mean=float(finite_values.mean(dtype=np.float64)),
        standard_deviation=float(finite_values.std(dtype=np.float64)),
        sum=float(finite_values.sum(dtype=np.float64)),
        negative_voxels=int(np.count_nonzero(finite_values < 0)),
        zero_voxels=int(np.count_nonzero(finite_values == 0)),
    )


def _downsample_yx_mean(array_zyx: np.ndarray, factor: int = 2) -> np.ndarray:
    """Mean-pool only Y and X while preserving all Z slices."""
    z_size, y_size, x_size = array_zyx.shape
    y_trim = y_size - y_size % factor
    x_trim = x_size - x_size % factor
    if y_trim < factor or x_trim < factor:
        raise ValueError(f"Cannot downsample spatial shape {(y_size, x_size)} by {factor}")
    cropped = array_zyx[:, :y_trim, :x_trim]
    return cropped.reshape(z_size, y_trim // factor, factor, x_trim // factor, factor).mean(axis=(2, 4), dtype=np.float32)


def _axes_descriptors(axes: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for axis in axes:
        item = {"name": axis, "type": "channel" if axis == "c" else "space"}
        if axis in "zyx":
            item["unit"] = "micrometer"
        output.append(item)
    return output


def _prepare_source_metadata_for_derived_store(
    source_attrs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate source metadata from attributes owned by the derived store.

    Attributes such as ``pft_level0_dtype`` describe the array currently stored
    at level 0. A deconvolution output is float32 even when the source is uint16,
    so copying the source value under the same key would be semantically wrong.
    The source values are retained together under
    ``pft_source_derived_root_metadata`` and all non-conflicting attributes keep
    their original keys.
    """
    direct_attrs = dict(source_attrs)
    derived_snapshot: dict[str, Any] = {}
    for key in sorted(_OUTPUT_OWNED_ROOT_ATTRS):
        if key in direct_attrs:
            derived_snapshot[key] = direct_attrs.pop(key)

    existing_snapshot = direct_attrs.pop(_SOURCE_DERIVED_METADATA_KEY, None)
    if existing_snapshot is not None:
        derived_snapshot["upstream_snapshot"] = existing_snapshot

    if derived_snapshot:
        direct_attrs[_SOURCE_DERIVED_METADATA_KEY] = derived_snapshot
    return direct_attrs, derived_snapshot


def _metadata_mismatches(
    root: zarr.Group,
    *,
    direct_source_attrs: Mapping[str, Any],
) -> list[str]:
    """Return source metadata fields not retained in the stored output."""
    mismatches: list[str] = []
    for key, expected_value in direct_source_attrs.items():
        actual_value = root.attrs.get(key)
        if actual_value != expected_value:
            mismatches.append(key)
    return mismatches


def _create_output_store(
    *,
    out_zarr: Path,
    shape_czyx: tuple[int, int, int, int],
    chunks_czyx: tuple[int, int, int, int],
    base_scale: Sequence[float],
    pyramid_max_layer: int,
    source_attrs: Mapping[str, Any],
    processing: Mapping[str, Any],
) -> tuple[zarr.Group, list[zarr.Array]]:
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out_zarr), mode="w")
    c_size, z_size, y_size, x_size = shape_czyx
    arrays: list[zarr.Array] = []
    datasets: list[dict[str, Any]] = []
    for level in range(pyramid_max_layer + 1):
        factor = 2**level
        shape = (c_size, z_size, max(1, y_size // factor), max(1, x_size // factor))
        chunks = (
            min(chunks_czyx[0], shape[0]),
            min(chunks_czyx[1], shape[1]),
            min(chunks_czyx[2], shape[2]),
            min(chunks_czyx[3], shape[3]),
        )
        arrays.append(root.create_dataset(str(level), shape=shape, chunks=chunks, dtype=np.float32, overwrite=True))
        level_scale = [
            float(value) * factor if axis in "yx" else float(value)
            for axis, value in zip("czyx", base_scale)
        ]
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{"type": "scale", "scale": level_scale}],
        })
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "image",
        "axes": _axes_descriptors("czyx"),
        "datasets": datasets,
    }]
    for key, value in source_attrs.items():
        root.attrs[key] = value
    root.attrs["pft_axes"] = "czyx"
    root.attrs["pft_level0_shape"] = list(shape_czyx)
    root.attrs["pft_level0_dtype"] = "float32"
    root.attrs["pft_multiscale_enabled"] = pyramid_max_layer > 0
    root.attrs["pft_pyramid_max_layer"] = int(pyramid_max_layer)
    root.attrs["pft_pyramid_downscale"] = 2
    root.attrs["pft_processing"] = dict(processing)
    return root, arrays


def _robust_limits(reference: np.ndarray) -> tuple[float, float]:
    values = np.asarray(reference, dtype=np.float32)
    low, high = np.percentile(values[np.isfinite(values)], (1.0, 99.8))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _rgb_destination_for_wavelength(wavelength_nm: float) -> tuple[int, str]:
    """Return RGB destination index and display name for one channel wavelength."""
    references = ((405.0, 2, "blue"), (488.0, 1, "green"), (561.0, 0, "red"))
    reference, destination, color = min(
        references,
        key=lambda item: abs(float(wavelength_nm) - item[0]),
    )
    if abs(float(wavelength_nm) - reference) > 20.0:
        raise ValueError(f"Cannot assign display color to wavelength {wavelength_nm:g} nm")
    return destination, color


def _rgb_composite(
    cyx: np.ndarray,
    limits: Sequence[tuple[float, float]],
    destinations: Sequence[int],
) -> np.ndarray:
    c_size, y_size, x_size = cyx.shape
    output = np.zeros((y_size, x_size, 3), dtype=np.float32)
    if len(destinations) < c_size:
        raise ValueError("RGB destination mapping is incomplete")
    for channel in range(min(c_size, 3)):
        low, high = limits[channel]
        output[..., destinations[channel]] = np.clip(
            (cyx[channel] - low) / (high - low), 0.0, 1.0
        )
    return output


def _exact_shared_limits(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """Return exact common display limits without allocating a combined image."""
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    if not np.isfinite(before).any() or not np.isfinite(after).any():
        raise ValueError("QC images contain no finite values")
    low = min(float(np.nanmin(before)), float(np.nanmin(after)))
    high = max(float(np.nanmax(before)), float(np.nanmax(after)))
    if high <= low:
        high = low + 1.0
    return low, high


def _normalize_with_limits(image: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    return np.clip((np.asarray(image, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


def save_selected_slice_qc(
    *,
    original_zarr: Path,
    original_level: int,
    deconvolved_zarr: Path,
    output_dir: Path,
    slices_1based: Sequence[int],
) -> list[Path]:
    """Save raw and normalized before/after QC without difference images.

    For each assigned slice, two PNGs are produced. The raw panel uses identical
    numeric display limits before and after for each channel. The normalized
    panel uses independent P1-P99.8 limits for each image and contains the merged
    RGB view plus all wavelength-mapped channels. Stored OME-Zarr values are not
    modified.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_meta = extract_ome_zarr_meta_for_compare(original_zarr, level=original_level)
    raw_root = zarr.open_group(str(original_zarr), mode="r")
    raw = raw_root[str(raw_meta["array_path"])]
    dec_root = zarr.open_group(str(deconvolved_zarr), mode="r")
    dec = dec_root["0"]
    if raw.shape != dec.shape:
        raise ValueError(f"QC shape mismatch: original={raw.shape}, deconvolved={dec.shape}")
    c_size, z_size, _, _ = raw.shape
    saved: list[Path] = []
    channel_names = raw_meta.get("channel_names") or [f"channel {index}" for index in range(c_size)]
    optics = resolve_channel_optics(original_zarr, level=original_level)
    if len(optics) != c_size:
        raise ValueError(f"Resolved {len(optics)} channel colors for C={c_size}")
    color_mapping = [_rgb_destination_for_wavelength(item.wavelength_nm) for item in optics]
    rgb_destinations = [item[0] for item in color_mapping]
    channel_labels = [
        f"{channel_names[index]} ({color_mapping[index][1]}, {optics[index].wavelength_nm:g} nm)"
        for index in range(c_size)
    ]

    for slice_number in slices_1based:
        z_index = int(slice_number) - 1
        if not 0 <= z_index < z_size:
            raise ValueError(f"QC slice Z{slice_number} is outside a stack with Z={z_size}")
        raw_cyx = np.asarray(raw[:, z_index], dtype=np.float32)
        dec_cyx = np.asarray(dec[:, z_index], dtype=np.float32)

        shared_limits = [
            _exact_shared_limits(raw_cyx[channel], dec_cyx[channel])
            for channel in range(c_size)
        ]
        raw_rgb = _rgb_composite(raw_cyx, shared_limits, rgb_destinations)
        dec_rgb = _rgb_composite(dec_cyx, shared_limits, rgb_destinations)

        rows = c_size + 1
        figure, axes = plt.subplots(rows, 2, figsize=(10, 3.6 * rows), constrained_layout=True)
        if rows == 1:
            axes = axes[None, :]
        axes[0, 0].imshow(raw_rgb)
        axes[0, 1].imshow(dec_rgb)
        axes[0, 0].set_ylabel("Merged RGB\n(shared raw limits)")
        for channel in range(c_size):
            row = channel + 1
            low, high = shared_limits[channel]
            before_image = axes[row, 0].imshow(raw_cyx[channel], cmap="gray", vmin=low, vmax=high)
            after_image = axes[row, 1].imshow(dec_cyx[channel], cmap="gray", vmin=low, vmax=high)
            figure.colorbar(before_image, ax=axes[row, 0], fraction=0.046)
            figure.colorbar(after_image, ax=axes[row, 1], fraction=0.046)
            axes[row, 0].set_ylabel(channel_labels[channel])
        axes[0, 0].set_title("Raw before deconvolution")
        axes[0, 1].set_title("Raw after deconvolution")
        for axis in axes.ravel():
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"Z{slice_number:03d}: raw Richardson-Lucy before/after "
            "(identical limits per channel)"
        )
        raw_output = output_dir / f"z{slice_number:03d}__raw_before_after.png"
        figure.savefig(raw_output, dpi=160)
        plt.close(figure)
        saved.append(raw_output)

        raw_norm_limits = [_robust_limits(raw_cyx[channel]) for channel in range(c_size)]
        dec_norm_limits = [_robust_limits(dec_cyx[channel]) for channel in range(c_size)]
        raw_norm = np.stack(
            [_normalize_with_limits(raw_cyx[channel], raw_norm_limits[channel]) for channel in range(c_size)]
        )
        dec_norm = np.stack(
            [_normalize_with_limits(dec_cyx[channel], dec_norm_limits[channel]) for channel in range(c_size)]
        )
        zero_one_limits = [(0.0, 1.0)] * c_size
        raw_norm_rgb = _rgb_composite(raw_norm, zero_one_limits, rgb_destinations)
        dec_norm_rgb = _rgb_composite(dec_norm, zero_one_limits, rgb_destinations)

        figure, axes = plt.subplots(rows, 2, figsize=(10, 3.6 * rows), constrained_layout=True)
        if rows == 1:
            axes = axes[None, :]
        axes[0, 0].imshow(raw_norm_rgb)
        axes[0, 1].imshow(dec_norm_rgb)
        axes[0, 0].set_ylabel("Merged RGB")
        for channel in range(c_size):
            row = channel + 1
            axes[row, 0].imshow(raw_norm[channel], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, 1].imshow(dec_norm[channel], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, 0].set_ylabel(channel_labels[channel])
        axes[0, 0].set_title("Normalized before deconvolution")
        axes[0, 1].set_title("Normalized after deconvolution")
        for axis in axes.ravel():
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"Z{slice_number:03d}: independently normalized before/after "
            "(P1-P99.8 per channel)"
        )
        normalized_output = output_dir / f"z{slice_number:03d}__normalized_before_after.png"
        figure.savefig(normalized_output, dpi=160)
        plt.close(figure)
        saved.append(normalized_output)
    return saved


def _write_reports(
    *,
    run_info: Mapping[str, Any],
    stats: Sequence[VolumeStats],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    stats_csv = output_dir / "deconvolution_channel_statistics.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(stats[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in stats)

    by_channel: dict[int, dict[str, VolumeStats]] = {}
    for item in stats:
        by_channel.setdefault(item.channel_index, {})[item.stage] = item
    checks: list[dict[str, Any]] = []
    for channel, stages in sorted(by_channel.items()):
        before = stages["input"]
        after = stages["deconvolved"]
        checks.append({
            "channel_index": channel,
            "channel_name": before.channel_name,
            "shape_and_voxel_count_preserved": before.voxel_count == after.voxel_count,
            "output_finite": after.finite_fraction == 1.0,
            "output_nonnegative": after.negative_voxels == 0,
            "stored_output_not_01_normalized": not (after.minimum >= 0.0 and after.maximum <= 1.0 and before.maximum > 1.0),
            "input_sum": before.sum,
            "output_sum": after.sum,
            "sum_ratio_output_to_input": after.sum / before.sum if before.sum else None,
            "input_min_max": [before.minimum, before.maximum],
            "output_min_max": [after.minimum, after.maximum],
        })

    report_data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": dict(run_info),
        "checks": checks,
        "statistics": [asdict(item) for item in stats],
    }
    report_json = output_dir / "deconvolution_validation.json"
    report_json.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    passed = all(
        check["shape_and_voxel_count_preserved"]
        and check["output_finite"]
        and check["output_nonnegative"]
        and check["stored_output_not_01_normalized"]
        for check in checks
    )
    lines = [
        "PFT 3D Richardson-Lucy validation report",
        "=" * 80,
        f"Status: {'PASS' if passed else 'FAIL'}",
        f"Generated (UTC): {report_data['generated_utc']}",
        f"Input OME-Zarr: {run_info['input_omezarr']}",
        f"Output OME-Zarr: {run_info['output_omezarr']}",
        f"Input level: {run_info['input_level']}",
        f"PSF model: {run_info['psf_model']}",
        "PSF policy: reusable level-0 master set, matched by wavelength metadata",
        f"Default iterations: {run_info.get('iterations_default', run_info.get('iterations'))}",
        "Iterations by channel:",
        *[
            "  C{channel_index} {channel_name} ({display_color}, {wavelength_nm:g} nm): {iterations}".format(**item)
            for item in run_info.get("iterations_by_channel", [])
        ],
        f"Source metadata preserved: {run_info.get('metadata_preserved_for_segmentation')}",
        "Stored numeric normalization: NONE",
        "Stored dtype: float32",
        "",
    ]
    for check in checks:
        lines.extend([
            f"Channel {check['channel_index']} ({check['channel_name']}):",
            f"  finite: {check['output_finite']}",
            f"  non-negative: {check['output_nonnegative']}",
            f"  not accidentally normalized to 0-1: {check['stored_output_not_01_normalized']}",
            f"  sum ratio output/input: {check['sum_ratio_output_to_input']}",
            f"  input min/max: {check['input_min_max']}",
            f"  output min/max: {check['output_min_max']}",
            "",
        ])
    report_txt = output_dir / "deconvolution_validation.txt"
    report_txt.write_text("\n".join(lines), encoding="utf-8")
    if not passed:
        raise ValueError(f"Deconvolution output failed validation. See {report_txt}")
    return report_txt, report_json, stats_csv


def deconvolve_omezarr_3ch_to_omezarr_skimage(
    *,
    in_omezarr: Path,
    out_root: Path,
    model: PSFModel = "BW",
    iters: int = 5,
    channel_iterations: Mapping[str, int] | None = None,
    background: float = 0.0,
    level: int = 0,
    channel_wavelength_nm: Mapping[str, float] | None = None,
    overwrite: bool = True,
    clip: bool = False,
    filter_epsilon: float | None = None,
    pyramid_max_layer: int = 2,
    preview_slices_1based: Sequence[int] | None = None,
) -> SkimageDeconvRunInfo:
    """Run channel-wise 3D Richardson-Lucy and save a validated multiscale output.

    ``clip`` must remain ``False`` for raw fluorescence values. The selected input
    level becomes level 0 of the derived OME-Zarr, and its physical voxel size is
    retained. Additional output levels are generated by lateral mean pooling.
    ``channel_iterations`` overrides the default ``iters`` value by channel color,
    wavelength, channel name, or channel index.
    """
    in_omezarr = Path(in_omezarr).resolve()
    out_root = Path(out_root).resolve()
    if preview_slices_1based is None:
        preview_slices_1based = (target_slice_for_volume(in_omezarr),)
    preview_slices_1based = tuple(sorted({int(value) for value in preview_slices_1based}))
    if not preview_slices_1based:
        raise ValueError("At least one QC slice is required")
    if int(iters) < 1:
        raise ValueError("iters must be at least 1")
    if clip:
        raise ValueError("clip=True is prohibited because it would clip raw fluorescence intensities to [-1, 1]")
    if background < 0:
        raise ValueError("background must be non-negative")
    if pyramid_max_layer < 0:
        raise ValueError("pyramid_max_layer must be non-negative")

    if level != MASTER_LEVEL:
        raise ValueError(
            "The reusable master PSFs are defined only for OME-Zarr level 0; "
            f"received level={level}"
        )
    meta = extract_ome_zarr_meta_for_compare(in_omezarr, level=level)
    axes = str(meta.get("axes") or "").lower()
    if axes != "czyx":
        raise ValueError(f"Expected CZYX input, got axes={axes!r}")
    root_in = zarr.open_group(str(in_omezarr), mode="r")
    array_in = root_in[str(meta["array_path"])]
    if array_in.ndim != 4:
        raise ValueError(f"Expected 4D CZYX array, got shape={array_in.shape}")
    c_size, z_size, y_size, x_size = (int(value) for value in array_in.shape)

    project_root = find_project_root(Path(__file__).resolve())
    master_matches, master_metadata = validate_stack_against_master_psfs(
        project_root=project_root,
        zarr_dir=in_omezarr,
        level=level,
        model=model,
        channel_wavelength_nm=channel_wavelength_nm,
        normalize_psfs=True,
    )
    matches_by_index = {match.channel_index: match for match in master_matches}
    if len(matches_by_index) != c_size:
        raise ValueError(f"Resolved {len(matches_by_index)} master PSF mappings for C={c_size}")
    iterations_by_index = _resolve_channel_iterations(
        master_matches,
        default_iters=int(iters),
        overrides=channel_iterations,
    )
    iteration_records = [
        {
            "channel_index": int(match.channel_index),
            "channel_name": str(match.channel_name),
            "wavelength_nm": float(match.wavelength_nm),
            "display_color": str(match.display_color),
            "iterations": int(iterations_by_index[match.channel_index]),
        }
        for match in master_matches
    ]
    iteration_tag = _iteration_tag(master_matches, iterations_by_index)

    relative_volume = relative_volume_path(in_omezarr)
    sample = relative_volume.name
    out_dir = (
        out_root
        / relative_volume.parent
        / f"{sample}__SK_RL__PSF{model}__{iteration_tag}__sourceL{level}"
    )
    out_zarr = out_dir / "image.ome.zarr"
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight = check_3d_sample(
        in_omezarr,
        level=level,
        require_masks=False,
    )
    preflight_report, _ = write_readiness_report(preflight, out_dir / "preflight")
    if not preflight.passed:
        raise ValueError(f"Input OME-Zarr failed pre-deconvolution checks: {preflight_report}")

    source_attrs_raw = copyable_root_metadata(in_omezarr)
    source_attrs, source_derived_metadata = _prepare_source_metadata_for_derived_store(
        source_attrs_raw
    )
    base_scale = coordinate_scale_for_level(in_omezarr, level=level)
    chunks_in = tuple(int(value) for value in (array_in.chunks or (1, 1, 256, 256)))
    processing = {
        "operation": "3d_richardson_lucy_deconvolution",
        "software": "skimage.restoration.richardson_lucy",
        "source_omezarr": str(in_omezarr),
        "source_level": int(level),
        "source_array_path": str(meta["array_path"]),
        "psf_model": model,
        "iterations": int(iters) if len(set(iterations_by_index.values())) == 1 else None,
        "iterations_default": int(iters),
        "iterations_by_channel": iteration_records,
        "background_subtracted": float(background),
        "clip": False,
        "filter_epsilon": filter_epsilon,
        "stored_dtype": "float32",
        "normalization_applied_to_stored_intensities": False,
        "output_level0_represents_source_level": int(level),
        "master_psf_reference_omezarr": master_metadata.get("reference_omezarr"),
        "master_psf_metadata_schema": master_metadata.get("schema"),
        "master_psf_channel_matching": "wavelength metadata",
        "master_psf_compatibility_warnings": list(
            master_metadata.get("compatibility_warnings") or []
        ),
        "resolved_optical_metadata_for_psf_compatibility": master_metadata.get(
            "resolved_optical_metadata_for_compatibility"
        ),
        "channel_psf_mapping": [asdict(item) | {"psf_path": str(item.psf_path)} for item in master_matches],
        "channel_display_colors": [
            {
                "channel_index": item.channel_index,
                "channel_name": item.channel_name,
                "wavelength_nm": item.wavelength_nm,
                "display_color": item.display_color,
                "display_color_hex": item.display_color_hex,
            }
            for item in master_matches
        ],
        "source_root_metadata_preserved": True,
        "source_output_owned_metadata_preserved_under": (
            _SOURCE_DERIVED_METADATA_KEY if source_derived_metadata else None
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_root_group, output_arrays = _create_output_store(
        out_zarr=out_zarr,
        shape_czyx=(c_size, z_size, y_size, x_size),
        chunks_czyx=chunks_in,
        base_scale=base_scale,
        pyramid_max_layer=pyramid_max_layer,
        source_attrs=source_attrs,
        processing=processing,
    )
    initial_metadata_mismatches = _metadata_mismatches(
        output_root_group,
        direct_source_attrs=source_attrs,
    )
    if initial_metadata_mismatches:
        raise ValueError(
            "Deconvolved OME-Zarr could not initialize preserved source metadata fields: "
            + ", ".join(initial_metadata_mismatches)
        )

    statistics: list[VolumeStats] = []
    for channel_index in range(c_size):
        match = matches_by_index[channel_index]
        channel_name = match.channel_name
        psf = _load_normalized_psf(match)
        volume_raw = np.asarray(array_in[channel_index], dtype=np.float32)
        prepared = _prepare_image_for_rl(volume_raw, background=background)
        statistics.append(_volume_stats("input", channel_index, channel_name, prepared))
        if any(psf_size > image_size for psf_size, image_size in zip(psf.shape, prepared.shape)):
            raise ValueError(f"PSF {psf.shape} is larger than channel volume {prepared.shape}")
        print(
            f"[DECONV] channel {channel_index}: {channel_name}; "
            f"wavelength={match.wavelength_nm:g} nm; color={match.display_color}"
        )
        channel_iters = int(iterations_by_index[channel_index])
        print(
            f"         input={prepared.shape} {prepared.dtype}; "
            f"master PSF={match.psf_path.name} {psf.shape}; iterations={channel_iters}"
        )
        deconvolved = np.asarray(
            richardson_lucy(
                image=prepared,
                psf=psf,
                num_iter=channel_iters,
                clip=False,
                filter_epsilon=filter_epsilon,
            ),
            dtype=np.float32,
        )
        if deconvolved.shape != prepared.shape:
            raise ValueError(f"Output shape mismatch: {deconvolved.shape} vs {prepared.shape}")
        statistics.append(_volume_stats("deconvolved", channel_index, channel_name, deconvolved))
        output_arrays[0][channel_index] = np.ascontiguousarray(deconvolved)
        level_data = deconvolved
        for output_level in range(1, pyramid_max_layer + 1):
            level_data = _downsample_yx_mean(level_data, factor=2)
            expected_shape = output_arrays[output_level].shape[1:]
            # Odd dimensions are trimmed by mean pooling; output arrays follow the same floor rule.
            if level_data.shape != expected_shape:
                level_data = level_data[:, : expected_shape[1], : expected_shape[2]]
            output_arrays[output_level][channel_index] = np.ascontiguousarray(level_data, dtype=np.float32)

    # Reopen from disk before validation so reports describe stored values.
    stored_root = zarr.open_group(str(out_zarr), mode="r")
    stored_level0 = stored_root["0"]
    metadata_mismatches = _metadata_mismatches(
        stored_root,
        direct_source_attrs=source_attrs,
    )
    if metadata_mismatches:
        raise ValueError(
            "Deconvolved OME-Zarr did not preserve source metadata fields: "
            + ", ".join(metadata_mismatches)
        )
    stored_stats: list[VolumeStats] = []
    for channel_index in range(c_size):
        stored = np.asarray(stored_level0[channel_index], dtype=np.float32)
        stored_stats.append(
            _volume_stats(
                "stored_deconvolved",
                channel_index,
                matches_by_index[channel_index].channel_name,
                stored,
            )
        )
        reference = next(item for item in statistics if item.stage == "deconvolved" and item.channel_index == channel_index)
        if not np.allclose(
            [reference.minimum, reference.maximum, reference.mean, reference.sum],
            [stored_stats[-1].minimum, stored_stats[-1].maximum, stored_stats[-1].mean, stored_stats[-1].sum],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(f"Stored-value check failed for channel {channel_index}")

    preview_dir = out_dir / "qc_target_slice_before_after"
    save_selected_slice_qc(
        original_zarr=in_omezarr,
        original_level=level,
        deconvolved_zarr=out_zarr,
        output_dir=preview_dir,
        slices_1based=preview_slices_1based,
    )

    run_data = {
        "input_omezarr": str(in_omezarr),
        "output_omezarr": str(out_zarr),
        "input_level": int(level),
        "output_level0_shape": list(stored_level0.shape),
        "output_levels": pyramid_max_layer + 1,
        "psf_model": model,
        "iterations": int(iters) if len(set(iterations_by_index.values())) == 1 else None,
        "iterations_default": int(iters),
        "iterations_by_channel": iteration_records,
        "background": float(background),
        "clip": False,
        "filter_epsilon": filter_epsilon,
        "normalization_applied_to_stored_intensities": False,
        "metadata_preserved_for_segmentation": True,
        "preserved_source_metadata_fields": sorted(source_attrs),
        "master_psf_mapping": [
            {
                "channel_index": item.channel_index,
                "channel_name": item.channel_name,
                "wavelength_nm": item.wavelength_nm,
                "display_color": item.display_color,
                "psf_path": str(item.psf_path),
            }
            for item in master_matches
        ],
        "preview_slices_1based": list(preview_slices_1based),
    }
    report_txt, report_json, stats_csv = _write_reports(
        run_info=run_data,
        stats=statistics,
        output_dir=out_dir,
    )
    return SkimageDeconvRunInfo(
        in_zarr=in_omezarr,
        out_zarr=out_zarr,
        out_dir=out_dir,
        model=model,
        iters=int(iters),
        iterations_by_channel={
            str(record["display_color"]): int(record["iterations"])
            for record in iteration_records
        },
        background=float(background),
        input_level=int(level),
        pyramid_max_layer=int(pyramid_max_layer),
        out_dtype="float32",
        clip=False,
        filter_epsilon=filter_epsilon,
        report_txt=report_txt,
        report_json=report_json,
        stats_csv=stats_csv,
        preview_dir=preview_dir,
        preflight_report=preflight_report,
    )


__all__ = [
    "PSFModel",
    "SkimageDeconvRunInfo",
    "VolumeStats",
    "deconvolve_omezarr_3ch_to_omezarr_skimage",
    "save_selected_slice_qc",
]
