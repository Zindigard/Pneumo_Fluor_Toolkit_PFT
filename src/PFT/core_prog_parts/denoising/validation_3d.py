"""Readiness checks for the PFT 3D restoration and 2.5D U-Net workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.metadata_3d import resolve_channel_optics

DEFAULT_TRAINING_SLICES_1BASED: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35)


@dataclass(frozen=True)
class CheckItem:
    """One PASS, WARN, or FAIL item in a 3D readiness report."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """Complete readiness result for one 3D sample."""

    sample: str
    image_zarr: str
    level: int
    checks: tuple[CheckItem, ...]

    @property
    def passed(self) -> bool:
        return not any(item.status == "FAIL" for item in self.checks)


def find_slice_mask(mask_sample_dir: Path, slice_1based: int) -> Path | None:
    """Find one binary target-slice mask using supported standard names."""
    names = (
        f"z{slice_1based:03d}_mask.tif",
        f"z{slice_1based:03d}_mask.tiff",
        f"mask_z{slice_1based:03d}.tif",
        f"mask_z{slice_1based:03d}.tiff",
        f"z{slice_1based:03d}.tif",
        f"z{slice_1based:03d}.tiff",
    )
    for name in names:
        path = mask_sample_dir / name
        if path.is_file():
            return path
    return None


def _small_spatial_sample(array: zarr.Array, axes: str) -> np.ndarray:
    index: list[Any] = []
    for axis, size_value in zip(axes, array.shape):
        size = int(size_value)
        if axis in "yx":
            index.append(slice(0, size, max(1, size // 64)))
        elif axis == "c":
            index.append(slice(None))
        elif axis == "z":
            index.append(min(size - 1, size // 2))
        else:
            index.append(0)
    return np.asarray(array[tuple(index)])


def _check_validation_report(image_zarr: Path) -> CheckItem:
    report = image_zarr.parent / "ome_zarr_validation.txt"
    if not report.is_file():
        return CheckItem("source OME-Zarr validation report", "WARN", f"missing: {report}")
    text = report.read_text(encoding="utf-8", errors="replace")
    status = "PASS" if "Status: PASS" in text else "FAIL"
    return CheckItem("source OME-Zarr validation report", status, str(report))


def check_3d_sample(
    image_zarr: str | Path,
    *,
    level: int,
    mask_root: str | Path | None = None,
    training_slices_1based: Sequence[int] = DEFAULT_TRAINING_SLICES_1BASED,
    expected_channels: int = 3,
    require_masks: bool = True,
) -> ReadinessReport:
    """Validate one 3D OME-Zarr and its sparse manual training masks."""
    image_zarr = Path(image_zarr).resolve()
    sample = image_zarr.parent.name
    checks: list[CheckItem] = []
    if not image_zarr.is_dir():
        return ReadinessReport(sample, str(image_zarr), level, (CheckItem("OME-Zarr exists", "FAIL", str(image_zarr)),))

    try:
        meta = extract_ome_zarr_meta_for_compare(image_zarr, level=level)
        root = zarr.open_group(str(image_zarr), mode="r")
        array_path = str(meta["array_path"])
        array = root[array_path]
    except Exception as error:
        return ReadinessReport(sample, str(image_zarr), level, (CheckItem("OME-Zarr readable", "FAIL", str(error)),))

    axes = str(meta.get("axes") or "").lower()
    shape = tuple(int(value) for value in meta.get("shape") or ())
    checks.append(CheckItem("OME-Zarr readable", "PASS", f"array={array_path}; shape={shape}; dtype={array.dtype}"))
    checks.append(CheckItem("axis order", "PASS" if axes == "czyx" else "FAIL", f"expected=czyx; stored={axes}"))

    if axes == "czyx" and len(shape) == 4:
        channel_count, z_count, y_size, x_size = shape
        checks.append(CheckItem("channel count", "PASS" if channel_count == expected_channels else "FAIL", f"expected={expected_channels}; stored={channel_count}"))
        checks.append(CheckItem("Z depth", "PASS" if z_count >= max(training_slices_1based) else "FAIL", f"Z={z_count}; largest requested training slice={max(training_slices_1based)}"))
        checks.append(CheckItem("spatial dimensions", "PASS" if y_size >= 64 and x_size >= 64 else "FAIL", f"YX={(y_size, x_size)}"))
    else:
        channel_count = z_count = y_size = x_size = 0

    voxel = meta.get("voxel_size_um") or {}
    voxel_ok = all(voxel.get(axis) is not None and float(voxel[axis]) > 0 for axis in "xyz")
    checks.append(CheckItem("voxel metadata", "PASS" if voxel_ok else "FAIL", str(voxel)))

    channel_names = meta.get("channel_names")
    names_ok = isinstance(channel_names, list) and len(channel_names) == channel_count
    checks.append(CheckItem("channel names", "PASS" if names_ok else "FAIL", repr(channel_names)))

    try:
        optics = resolve_channel_optics(image_zarr, level=level)
        detail = "; ".join(f"{item.index}:{item.name}={item.wavelength_nm:g} nm [{item.wavelength_source}]" for item in optics)
        checks.append(CheckItem("channel wavelengths", "PASS", detail))
    except Exception as error:
        checks.append(CheckItem("channel wavelengths", "FAIL", str(error)))

    checks.append(_check_validation_report(image_zarr))

    try:
        sample_values = _small_spatial_sample(array, axes)
        finite = bool(np.isfinite(sample_values).all())
        nonconstant = bool(sample_values.size and np.nanmin(sample_values) < np.nanmax(sample_values))
        status = "PASS" if finite and nonconstant else "FAIL"
        checks.append(CheckItem("representative values", status, f"shape={sample_values.shape}; finite={finite}; nonconstant={nonconstant}; min={float(np.nanmin(sample_values)):.6g}; max={float(np.nanmax(sample_values)):.6g}"))
    except Exception as error:
        checks.append(CheckItem("representative values", "FAIL", str(error)))

    if require_masks:
        if mask_root is None:
            checks.append(CheckItem("manual mask root", "FAIL", "mask_root was not provided"))
        else:
            mask_sample_dir = Path(mask_root) / sample
            checks.append(CheckItem("manual mask folder", "PASS" if mask_sample_dir.is_dir() else "FAIL", str(mask_sample_dir)))
            for slice_number in training_slices_1based:
                path = find_slice_mask(mask_sample_dir, slice_number)
                if path is None:
                    checks.append(CheckItem(f"manual mask Z{slice_number}", "FAIL", "missing binary TIFF"))
                    continue
                try:
                    mask = np.asarray(tiff.imread(str(path)))
                    if mask.ndim > 2:
                        mask = np.squeeze(mask)
                    shape_ok = mask.ndim == 2 and mask.shape == (y_size, x_size)
                    values = np.unique(mask)
                    binary_ok = bool(np.all(np.isin(values, (0, 1, 255))))
                    fg = int(np.count_nonzero(mask))
                    bg = int(mask.size - fg)
                    status = "PASS" if shape_ok and binary_ok and fg > 0 and bg > 0 else "FAIL"
                    checks.append(CheckItem(f"manual mask Z{slice_number}", status, f"path={path}; shape={mask.shape}; values={values[:8].tolist()}; foreground={fg}; background={bg}"))
                except Exception as error:
                    checks.append(CheckItem(f"manual mask Z{slice_number}", "FAIL", f"{path}: {error}"))

    return ReadinessReport(sample, str(image_zarr), level, tuple(checks))


def write_readiness_report(report: ReadinessReport, output_dir: str | Path) -> tuple[Path, Path]:
    """Write human-readable TXT and machine-readable JSON reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "PFT 3D processing readiness report",
        "=" * 80,
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Status: {status}",
        f"Sample: {report.sample}",
        f"OME-Zarr: {report.image_zarr}",
        f"Level: {report.level}",
        "",
        f"{'RESULT':<8} | CHECK | DETAIL",
        "-" * 80,
    ]
    for item in report.checks:
        lines.append(f"{item.status:<8} | {item.name} | {item.detail}")
    text_path = output_dir / f"{report.sample}__3d_input_check.txt"
    json_path = output_dir / f"{report.sample}__3d_input_check.json"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({**asdict(report), "passed": report.passed}, indent=2), encoding="utf-8")
    return text_path, json_path


__all__ = [
    "CheckItem",
    "DEFAULT_TRAINING_SLICES_1BASED",
    "ReadinessReport",
    "check_3d_sample",
    "find_slice_mask",
    "write_readiness_report",
]
