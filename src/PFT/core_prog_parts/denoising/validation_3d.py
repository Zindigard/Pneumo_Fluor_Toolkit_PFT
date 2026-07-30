"""Readiness checks for the PFT 3D restoration and 2.5D U-Net workflow."""

from __future__ import annotations

import hashlib
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

# One manually selected target slice per source volume. Keys are stable paths
# relative to results/img/3d_data and therefore remain unique when sample names
# occur in more than one acquisition directory.
TARGET_SLICE_BY_VOLUME_KEY: dict[str, int] = {
    "20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM": 6,
    "20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI2_SIM": 10,
    "20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI3_SIM": 16,
    "20220218_dynamic/WT_NHS_HADA_NADA_TADA_40min_ROI1_SIM": 13,
    "20220218_dynamic/WT_THY_HADA_NADA_TADA_40min_ROI1_SIM": 31,
    "20220218_dynamic/WT_THY_HADA_NADA_TADA_40min_ROI2_SIM": 15,
    "20220218_dynamic/WT_THY_HADA_NADA_TADA_40min_ROI3_SIM": 16,
    "20220225_HADA_NADA_TADA_40min/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM": 20,
    "20220225_HADA_NADA_TADA_40min/DpspA_THY_HADA_NADA_TADA_40min_ROI2_SIM": 18,
    "20220225_HADA_NADA_TADA_40min/DpspA_THY_HADA_NADA_TADA_40min_ROI3_SIM": 17,
    "20220225_HADA_NADA_TADA_40min/WT_NHS_HADA_NADA_TADA_40min_ROI1_SIM": 10,
    "20220225_HADA_NADA_TADA_40min/WT_NHS_HADA_NADA_TADA_40min_ROI2_SIM": 32,
    "20220225_HADA_NADA_TADA_40min/WT_NHS_HADA_NADA_TADA_40min_ROI3_SIM": 19,
    "20220225_HADA_NADA_TADA_40min/WT_THY_HADA_NADA_TADA_40min_ROI1_SIM": 6,
    "20220225_HADA_NADA_TADA_40min/WT_THY_HADA_NADA_TADA_40min_ROI2_SIM": 18,
    "20220318_HADA_NADA_TADA_40min/DpspA_NHS_NADA_HADA_TADA_40min_ROI1_SIM": 39,
    "20220318_HADA_NADA_TADA_40min/DpspA_NHS_NADA_HADA_TADA_40min_ROI2_SIM": 23,
    "20220318_HADA_NADA_TADA_40min/DpspA_NHS_NADA_HADA_TADA_40min_ROI3_SIM": 21,
    "20220318_HADA_NADA_TADA_40min/DpspA_NHS_NADA_HADA_TADA_40min_ROI4_SIM": 22,
    "20220318_HADA_NADA_TADA_40min/WT_NHS_NADA_HADA_TADA_40min_ROI1_SIM": 20,
    "20220318_HADA_NADA_TADA_40min/WT_NHS_NADA_HADA_TADA_40min_ROI2_SIM": 23,
    "20220318_HADA_NADA_TADA_40min/WT_NHS_NADA_HADA_TADA_40min_ROI3_SIM": 23,
    "20220330_HADA_NADA_TADA_0min/DpspA_NHS_HADA_NADA_TADA_0min_ROI1_SIM": 15,
    "20220330_HADA_NADA_TADA_0min/DpspA_NHS_HADA_NADA_TADA_0min_ROI2_SIM": 20,
    "20220330_HADA_NADA_TADA_0min/DpspA_THY_HADA_NADA_TADA_0min_ROI1_SIM": 21,
    "20220330_HADA_NADA_TADA_0min/WT_NHS_HADA_NADA_TADA_0min_ROI1_SIM": 7,
    "20220330_HADA_NADA_TADA_0min/WT_NHS_HADA_NADA_TADA_0min_ROI2_SIM": 16,
    "20220330_HADA_NADA_TADA_0min/WT_THY_HADA_NADA_TADA_0min_ROI1_SIM": 20,
    "20220330_HADA_NADA_TADA_0min/WT_THY_HADA_NADA_TADA_0min_ROI2_SIM": 39,
}

# Fixed deconvolution/PSF test cohort: one stack from each acquisition folder.
# The assigned target slices span shallow, middle, and deep optical planes.
DEFAULT_DECONV_TEST_VOLUME_KEYS: tuple[str, ...] = (
    "20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM",
    "20220225_HADA_NADA_TADA_40min/WT_NHS_HADA_NADA_TADA_40min_ROI1_SIM",
    "20220318_HADA_NADA_TADA_40min/DpspA_NHS_NADA_HADA_TADA_40min_ROI3_SIM",
    "20220330_HADA_NADA_TADA_0min/WT_THY_HADA_NADA_TADA_0min_ROI2_SIM",
)

# Retained only for backward-compatible imports. New 3D code must resolve the
# target from TARGET_SLICE_BY_VOLUME_KEY instead of assuming one global slice.
DEFAULT_TRAINING_SLICES_1BASED: tuple[int, ...] = (10,)


def relative_volume_path(image_zarr: str | Path, image_root: str | Path | None = None) -> Path:
    """Return a stable experiment/sample path for one source OME-Zarr.

    The preferred source layout is ``<image_root>/<experiment>/<sample>/image.ome.zarr``.
    When ``image_root`` is omitted, the nearest ancestor named ``3d_data`` is
    used. Falling back to the sample folder keeps the helper usable for custom
    external inputs.
    """
    image_zarr = Path(image_zarr).resolve()
    sample_dir = image_zarr.parent
    if image_root is not None:
        root = Path(image_root).resolve()
        try:
            relative = sample_dir.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"OME-Zarr {image_zarr} is not located below image root {root}"
            ) from error
        if not relative.parts:
            raise ValueError(f"OME-Zarr has no experiment/sample path below {root}: {image_zarr}")
        return relative

    for ancestor in sample_dir.parents:
        if ancestor.name == "3d_data":
            return sample_dir.relative_to(ancestor)
    return Path(sample_dir.name)


def volume_key(image_zarr: str | Path, image_root: str | Path | None = None) -> str:
    """Return a unique forward-slash experiment/sample identifier."""
    return relative_volume_path(image_zarr, image_root).as_posix()


def target_slice_for_volume(
    image_zarr: str | Path,
    image_root: str | Path | None = None,
) -> int:
    """Return the configured one-based target slice for one source volume."""
    key = volume_key(image_zarr, image_root)
    try:
        return int(TARGET_SLICE_BY_VOLUME_KEY[key])
    except KeyError as error:
        raise KeyError(
            f"No target slice is configured for {key!r}. Add the volume to "
            "TARGET_SLICE_BY_VOLUME_KEY before labeling, training, inference, or QC."
        ) from error


def target_context_for_volume(
    image_zarr: str | Path,
    image_root: str | Path | None = None,
) -> tuple[int, int, int]:
    """Return the one-based Z-1/Z/Z+1 context for the configured target."""
    target = target_slice_for_volume(image_zarr, image_root)
    return target - 1, target, target + 1


def configured_test_zarrs(image_root: str | Path) -> list[Path]:
    """Resolve the fixed four-stack deconvolution test cohort."""
    root = Path(image_root).resolve()
    paths: list[Path] = []
    missing: list[Path] = []
    for key in DEFAULT_DECONV_TEST_VOLUME_KEYS:
        path = root / Path(*key.split("/")) / "image.ome.zarr"
        if path.is_dir():
            paths.append(path)
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "The fixed four-stack test cohort is incomplete:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    return paths


def volume_file_id(image_zarr: str | Path, image_root: str | Path | None = None) -> str:
    """Return a filesystem-safe identifier for reports and QC filenames."""
    return "__".join(relative_volume_path(image_zarr, image_root).parts)


def annotation_sample_dir(
    mask_root: str | Path,
    image_zarr: str | Path,
    image_root: str | Path | None = None,
) -> Path:
    """Return the collision-free manual-mask folder for one source volume."""
    return Path(mask_root) / relative_volume_path(image_zarr, image_root)


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
    training_slices_1based: Sequence[int] | None = None,
    expected_channels: int = 3,
    require_masks: bool = True,
) -> ReadinessReport:
    """Validate one 3D OME-Zarr and its sparse manual training masks."""
    image_zarr = Path(image_zarr).resolve()
    sample = volume_key(image_zarr)
    if training_slices_1based is None:
        try:
            training_slices_1based = (target_slice_for_volume(image_zarr),)
        except KeyError as error:
            return ReadinessReport(
                sample,
                str(image_zarr),
                level,
                (CheckItem("configured target slice", "FAIL", str(error)),),
            )
    training_slices_1based = tuple(sorted({int(value) for value in training_slices_1based}))
    if not training_slices_1based:
        raise ValueError("At least one target slice is required")
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
        context_ok = all(2 <= value <= z_count - 1 for value in training_slices_1based)
        checks.append(
            CheckItem(
                "Z target context",
                "PASS" if context_ok else "FAIL",
                f"Z={z_count}; targets={tuple(training_slices_1based)}; each requires Z-1/Z/Z+1",
            )
        )
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
            mask_sample_dir = annotation_sample_dir(mask_root, image_zarr)
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
    # Keep report names compact because the deconvolution output directory can
    # already be long on Windows. A stable digest prevents collisions when
    # several reports are written into the same directory.
    report_id = hashlib.sha1(report.sample.encode("utf-8")).hexdigest()[:10]
    text_path = output_dir / f"3d_input_check_{report_id}.txt"
    json_path = output_dir / f"3d_input_check_{report_id}.json"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({**asdict(report), "passed": report.passed}, indent=2), encoding="utf-8")
    return text_path, json_path


__all__ = [
    "CheckItem",
    "DEFAULT_DECONV_TEST_VOLUME_KEYS",
    "DEFAULT_TRAINING_SLICES_1BASED",
    "TARGET_SLICE_BY_VOLUME_KEY",
    "ReadinessReport",
    "annotation_sample_dir",
    "check_3d_sample",
    "find_slice_mask",
    "configured_test_zarrs",
    "relative_volume_path",
    "target_context_for_volume",
    "target_slice_for_volume",
    "volume_file_id",
    "volume_key",
    "write_readiness_report",
]
