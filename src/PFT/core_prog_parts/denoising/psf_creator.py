"""Create and validate one reusable master PSF set for the 3D dataset.

The project assumes that all 3D stacks were acquired and reconstructed with the
same microscope configuration. Therefore, PSFs are generated once from one
reference level-0 OME-Zarr stack and then reused for every compatible stack.
Only the emission wavelength differs between the three fluorescence channels.

The master set contains exactly three normalized PSFs:

* approximately 405 nm, displayed as blue;
* approximately 488 nm, displayed as green;
* approximately 561 nm, displayed as red.

Every stack is validated against the reference metadata before a master PSF is
used. Processing stops when a present acquisition value conflicts with the
master reference. Some converted OME-Zarr stores omit refractive-index fields;
in that case the fixed master-reference values are reused with an explicit
warning. Channel-to-PSF assignment is performed by wavelength, not by channel
name or array order.

External software requirements
------------------------------
PSF generation requires:

* an ImageJ/Fiji installation containing Java and ImageJ JAR files;
* a PSF Generator folder or JAR file containing ``PSF_Generator.jar`` or
  ``PSFGenerator.jar``.

Pass both paths explicitly through ``scripts/denoising/create_psf.py`` when they
are not available in the project-managed Fiji installation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.fuji_managment import (
    ensure_fiji_in_project,
    ensure_psf_generator_exists,
)
from PFT.core_prog_parts.denoising.metadata_3d import (
    ChannelOptics,
    resolve_channel_optics,
)

ModelName = Literal["BW", "GL", "RW"]
STANDARD_CONFIG_BASENAME = "PSFGenerator_standard"
MASTER_METADATA_BASENAME = "psf_master_metadata.json"
MASTER_LEVEL = 0
NORMALIZATION_ATOL = 1e-5
DEFAULT_WAVELENGTH_TOLERANCE_NM = 2.0

# Stable scientific display convention used by the 3D workflow.
REFERENCE_WAVELENGTH_COLORS: tuple[tuple[float, str, str], ...] = (
    (405.0, "blue", "0000FF"),
    (488.0, "green", "00FF00"),
    (561.0, "red", "FF0000"),
)


@dataclass(frozen=True)
class MasterPSFJob:
    """One wavelength-specific member of the reusable master PSF set."""

    model: ModelName
    wavelength_nm: float
    wavelength_source: str
    display_color: str
    display_color_hex: str
    reference_channel_index: int
    reference_channel_name: str
    level: int
    out_tif: Path
    config_txt: Path


@dataclass(frozen=True)
class PSFValidation:
    """Validation result for one master PSF array."""

    suitable: bool
    normalized: bool
    psf_sum: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class MasterPSFMatch:
    """Validated mapping from one image channel to one master PSF."""

    channel_index: int
    channel_name: str
    wavelength_nm: float
    wavelength_source: str
    display_color: str
    display_color_hex: str
    psf_path: Path


def _available_levels(zarr_dir: Path) -> list[int]:
    """Return OME-Zarr pyramid-level indices."""
    root = zarr.open_group(str(zarr_dir), mode="r")
    multiscales = root.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return [0]
    datasets = multiscales[0].get("datasets")
    return list(range(len(datasets))) if isinstance(datasets, list) and datasets else [0]


def _prompt_level(levels: list[int], default: int = MASTER_LEVEL) -> int:
    """Prompt for a valid OME-Zarr level and enforce the master level."""
    if MASTER_LEVEL not in levels:
        raise ValueError(f"Reference OME-Zarr does not contain required level {MASTER_LEVEL}")
    answer = input(f"Choose level [{MASTER_LEVEL}]: ").strip()
    selected = MASTER_LEVEL if answer == "" else int(answer)
    if selected != MASTER_LEVEL:
        raise ValueError("The reusable master PSF set must be generated for OME-Zarr level 0")
    return selected


def _parse_psfgenerator_config(path: Path) -> dict[str, str]:
    """Parse a PSFGenerator key-value configuration file."""
    params: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def write_psfgenerator_config(
    params: Mapping[str, str],
    dst: Path,
    *,
    header: Iterable[str] = (),
) -> None:
    """Write a documented PSFGenerator configuration with deterministic ordering."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# " + text for text in header]
    if lines:
        lines.append("")
    preferred = [
        "Type", "Lambda", "NA", "NX", "NY", "NZ", "ResLateral",
        "ResAxial", "Scale", "unit",
    ]
    emitted: set[str] = set()
    for key in preferred:
        if key in params:
            lines.append(f"{key}={params[key]}")
            emitted.add(key)
    for key in sorted(params):
        if key not in emitted:
            lines.append(f"{key}={params[key]}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _builtin_standard_params(model: ModelName) -> dict[str, str]:
    """Return a conservative standard template if no previous template exists."""
    params = {
        "Type": model,
        "Lambda": "488",
        "NA": "1.4",
        "NX": "128",
        "NY": "128",
        "NZ": "40",
        "ResLateral": "40",
        "ResAxial": "110",
        "Scale": "Linear",
        "unit": "nm",
        f"psf-{model}-accuracy": "Best",
    }
    if model in ("BW", "RW"):
        params[f"psf-{model}-NI"] = "1.518"
    if model == "GL":
        params["psf-GL-NS"] = "1.33"
        params["psf-GL-NI"] = "1.518"
    return params


def ensure_standard_config(project_root: Path, model: ModelName) -> Path:
    """Create the standard named configuration before metadata substitution."""
    config_dir = project_root / "results" / "psf" / "configuration"
    standard = config_dir / f"{STANDARD_CONFIG_BASENAME}_{model}.txt"
    legacy = project_root / "results" / "psf" / f"config{model}.txt"
    if legacy.is_file():
        params = _parse_psfgenerator_config(legacy)
        source = str(legacy)
    elif standard.is_file():
        return standard
    else:
        params = _builtin_standard_params(model)
        source = "built-in documented fallback template"
    write_psfgenerator_config(
        params,
        standard,
        header=(
            "PFT standard PSFGenerator configuration.",
            f"Model: {model}",
            f"Template source: {source}",
            "Reference-image metadata are inserted into three master configurations.",
        ),
    )
    return standard


def _shape_zyx(meta: Mapping[str, Any]) -> tuple[int, int, int]:
    axes = str(meta.get("axes") or "").lower()
    shape = tuple(meta.get("shape") or ())
    if not all(axis in axes for axis in "zyx"):
        raise ValueError(f"Expected spatial ZYX axes, got axes={axes!r}")
    return tuple(int(shape[axes.index(axis)]) for axis in "zyx")


def update_params_for_image_and_channel(
    base_params: dict[str, str],
    *,
    img_meta: dict,
    wavelength_nm: float,
    model: ModelName,
    accuracy: str = "Best",
    match_sampling_to_image: bool = True,
    match_na_to_image: bool = True,
    match_refractive_indices: bool = True,
) -> dict[str, str]:
    """Insert reference sampling, dimensions, optics, and channel wavelength."""
    params = dict(base_params)
    params["Type"] = model
    params["Lambda"] = f"{float(wavelength_nm):.10g}"
    params["unit"] = "nm"

    z_size, y_size, x_size = _shape_zyx(img_meta)
    params["NZ"] = str(z_size)
    params.setdefault("NX", str(min(x_size, 256)))
    params.setdefault("NY", str(min(y_size, 256)))

    if match_na_to_image and img_meta.get("objective_na") is not None:
        params["NA"] = f"{float(img_meta['objective_na']):.10g}"

    if match_sampling_to_image:
        voxel = img_meta.get("voxel_size_um") or {}
        lateral_values = [value for value in (voxel.get("x"), voxel.get("y")) if value is not None]
        if lateral_values:
            params["ResLateral"] = f"{1000.0 * float(sum(lateral_values) / len(lateral_values)):.10g}"
        if voxel.get("z") is not None:
            params["ResAxial"] = f"{1000.0 * float(voxel['z']):.10g}"

    if match_refractive_indices:
        immersion_ri = img_meta.get("refractive_index_immersion")
        sample_ri = img_meta.get("refractive_index_sample")
        if immersion_ri is not None:
            params[f"psf-{model}-NI"] = f"{float(immersion_ri):.10g}"
        if model == "GL" and sample_ri is not None:
            params["psf-GL-NS"] = f"{float(sample_ri):.10g}"

    params[f"psf-{model}-accuracy"] = str(accuracy)
    return params


def _config_hash(params: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(sorted(params.items())), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _display_color_for_wavelength(wavelength_nm: float) -> tuple[str, str]:
    reference, color, color_hex = min(
        REFERENCE_WAVELENGTH_COLORS,
        key=lambda item: abs(float(wavelength_nm) - item[0]),
    )
    if abs(float(wavelength_nm) - reference) > 20.0:
        raise ValueError(
            f"Wavelength {wavelength_nm:g} nm cannot be assigned reliably to blue, green, or red"
        )
    return color, color_hex


def _canonical_wavelength_filename(wavelength_nm: float) -> str:
    return f"psf_{int(round(float(wavelength_nm)))}nm_L{MASTER_LEVEL}.tif"


def master_psf_directory(project_root: Path) -> Path:
    """Return the fixed storage location for the reusable master PSF set."""
    return project_root.resolve() / "results" / "psf" / "master"


def _reference_signature(
    *,
    zarr_dir: Path,
    img_meta: Mapping[str, Any],
    optics: Sequence[ChannelOptics],
    model: ModelName,
    accuracy: str,
) -> dict[str, Any]:
    channels: list[dict[str, Any]] = []
    for channel in optics:
        color, color_hex = _display_color_for_wavelength(channel.wavelength_nm)
        channels.append({
            "reference_channel_index": int(channel.index),
            "reference_channel_name": str(channel.name),
            "wavelength_nm": float(channel.wavelength_nm),
            "wavelength_source": str(channel.wavelength_source),
            "display_color": color,
            "display_color_hex": color_hex,
            "master_psf_filename": _canonical_wavelength_filename(channel.wavelength_nm),
        })
    channels.sort(key=lambda item: item["wavelength_nm"])
    return {
        "schema": "pft-master-psf-v1",
        "reference_omezarr": str(zarr_dir.resolve()),
        "reference_level": MASTER_LEVEL,
        "reference_array_path": img_meta.get("array_path"),
        "reference_axes": str(img_meta.get("axes") or "").lower(),
        "reference_shape_zyx": list(_shape_zyx(img_meta)),
        "voxel_size_um": img_meta.get("voxel_size_um"),
        "objective_na": _float_or_none(img_meta.get("objective_na")),
        "refractive_index_immersion": _float_or_none(img_meta.get("refractive_index_immersion")),
        "refractive_index_sample": _float_or_none(img_meta.get("refractive_index_sample")),
        "channel_count": len(optics),
        "channels": channels,
        "model": model,
        "accuracy": str(accuracy),
    }


def _assert_three_unique_wavelengths(optics: Sequence[ChannelOptics]) -> None:
    if len(optics) != 3:
        raise ValueError(f"The master workflow requires exactly 3 channels, found {len(optics)}")
    wavelengths = [float(item.wavelength_nm) for item in optics]
    for index, first in enumerate(wavelengths):
        for second in wavelengths[index + 1:]:
            if abs(first - second) <= DEFAULT_WAVELENGTH_TOLERANCE_NM:
                raise ValueError(f"Channel wavelengths are not unique: {wavelengths}")
    colors = [_display_color_for_wavelength(value)[0] for value in wavelengths]
    if set(colors) != {"blue", "green", "red"}:
        raise ValueError(
            f"Expected one blue, one green, and one red channel, resolved colors={colors}"
        )


def plan_master_psf_jobs(
    *,
    project_root: Path,
    reference_zarr: Path,
    model: ModelName = "BW",
    channel_wavelength_nm: Mapping[str, float] | None = None,
    accuracy: str = "Best",
) -> tuple[list[MasterPSFJob], dict[str, Any]]:
    """Create three reference-derived master configurations and output paths."""
    project_root = project_root.resolve()
    reference_zarr = reference_zarr.resolve()
    img_meta = extract_ome_zarr_meta_for_compare(reference_zarr, level=MASTER_LEVEL)
    axes = str(img_meta.get("axes") or "").lower()
    if axes != "czyx":
        raise ValueError(f"Expected reference axes CZYX at level 0, got {axes!r}")
    optics = resolve_channel_optics(
        reference_zarr,
        level=MASTER_LEVEL,
        explicit_wavelength_nm=channel_wavelength_nm,
    )
    _assert_three_unique_wavelengths(optics)

    master_dir = master_psf_directory(project_root)
    config_dir = master_dir / "configuration"
    master_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    standard_config = ensure_standard_config(project_root, model)
    standard_params = _parse_psfgenerator_config(standard_config)
    signature = _reference_signature(
        zarr_dir=reference_zarr,
        img_meta=img_meta,
        optics=optics,
        model=model,
        accuracy=accuracy,
    )

    jobs: list[MasterPSFJob] = []
    for channel in optics:
        color, color_hex = _display_color_for_wavelength(channel.wavelength_nm)
        params = update_params_for_image_and_channel(
            standard_params,
            img_meta=img_meta,
            wavelength_nm=channel.wavelength_nm,
            model=model,
            accuracy=accuracy,
        )
        rounded = int(round(channel.wavelength_nm))
        config_path = config_dir / f"PSFGenerator_master_{model}_{rounded}nm_L0.txt"
        output_path = master_dir / _canonical_wavelength_filename(channel.wavelength_nm)
        write_psfgenerator_config(
            params,
            config_path,
            header=(
                "PFT reusable master PSF configuration.",
                f"Reference OME-Zarr: {reference_zarr}",
                "Reference level: 0",
                f"Reference channel: {channel.name} (index {channel.index})",
                f"Wavelength: {channel.wavelength_nm:g} nm; source: {channel.wavelength_source}",
                f"Display color: {color} (#{color_hex})",
                f"Configuration SHA-256: {_config_hash(params)}",
                "This PSF is reused for every stack that passes master-metadata validation.",
            ),
        )
        jobs.append(MasterPSFJob(
            model=model,
            wavelength_nm=float(channel.wavelength_nm),
            wavelength_source=str(channel.wavelength_source),
            display_color=color,
            display_color_hex=color_hex,
            reference_channel_index=int(channel.index),
            reference_channel_name=str(channel.name),
            level=MASTER_LEVEL,
            out_tif=output_path,
            config_txt=config_path,
        ))
    jobs.sort(key=lambda item: item.wavelength_nm)
    return jobs, signature


def _find_java_exe(imagej_dir: Path) -> Path:
    candidates = [
        imagej_dir / "java" / "win64" / "bin" / "java.exe",
        imagej_dir / "java" / "bin" / "java.exe",
        imagej_dir / "jre" / "bin" / "java.exe",
        imagej_dir / "Contents" / "java" / "bin" / "java",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    nested = list(imagej_dir.glob("java/**/bin/java.exe")) + list(imagej_dir.glob("java/**/bin/java"))
    if nested:
        return nested[0]
    java_home = os.environ.get("JAVA_HOME")
    executable = "java.exe" if os.name == "nt" else "java"
    if java_home and (Path(java_home) / "bin" / executable).is_file():
        return Path(java_home) / "bin" / executable
    system_java = shutil.which("java")
    if system_java:
        return Path(system_java)
    raise FileNotFoundError("Java was not found in ImageJ/Fiji, JAVA_HOME, or PATH")


def _find_psf_jar(imagej_dir: Path, psf_creator_dir: Path | None) -> Path:
    candidates: list[Path] = []
    if psf_creator_dir is not None:
        if psf_creator_dir.is_file():
            candidates.append(psf_creator_dir)
        else:
            candidates.extend([
                psf_creator_dir / "PSF_Generator.jar",
                psf_creator_dir / "PSFGenerator.jar",
                *psf_creator_dir.glob("**/PSF_Generator.jar"),
                *psf_creator_dir.glob("**/PSFGenerator.jar"),
            ])
    candidates.extend([
        imagej_dir / "plugins" / "PSF_Generator.jar",
        imagej_dir / "plugins" / "PSFGenerator.jar",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "PSF Generator JAR was not found. Pass --psf-creator-dir with the folder "
        "or JAR path containing PSF_Generator.jar"
    )


def _normalize_psf_file_in_place(
    path: Path,
    *,
    res_lateral_nm: float | None = None,
    res_axial_nm: float | None = None,
) -> tuple[tuple[int, int, int], float]:
    """Normalize a 3D PSF to sum 1 and retain calibrated TIFF metadata."""
    array = np.asarray(tiff.imread(str(path)), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D PSF ZYX, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"PSF contains NaN or infinite values: {path}")
    array = np.maximum(array, 0.0)
    total = float(array.sum(dtype=np.float64))
    if total <= 0:
        raise ValueError(f"PSF has zero non-negative sum: {path}")
    array /= np.float32(total)

    metadata: dict[str, Any] = {
        "unit": "nm",
        "axes": "ZYX",
        "hyperstack": True,
        "mode": "grayscale",
        "channels": 1,
        "slices": int(array.shape[0]),
        "frames": 1,
    }
    write_kwargs: dict[str, Any] = {}
    if res_axial_nm is not None:
        metadata["spacing"] = float(res_axial_nm)
    if res_lateral_nm is not None and res_lateral_nm > 0:
        write_kwargs["resolution"] = (
            1.0 / float(res_lateral_nm),
            1.0 / float(res_lateral_nm),
        )
    temporary = path.with_name(path.stem + ".normalized.tmp.tif")
    tiff.imwrite(
        str(temporary),
        array,
        imagej=True,
        metadata=metadata,
        photometric="minisblack",
        **write_kwargs,
    )
    temporary.replace(path)
    return tuple(int(value) for value in array.shape), float(array.sum(dtype=np.float64))


def run_psfgenerator_cli_v2(
    *,
    start_path: Path,
    config_path: Path,
    out_path: Path,
    imagej_dir: Path | None = None,
    psf_creator_dir: Path | None = None,
    quiet: bool = False,
    expected_min_filesize_bytes: int = 1024,
) -> Path:
    """Run PSFGenerator and write a normalized calibrated master TIFF."""
    project_root = find_project_root(start_path.resolve())
    if imagej_dir is None:
        imagej_dir = ensure_fiji_in_project(project_root, quiet=quiet)
        ensure_psf_generator_exists(imagej_dir, auto_download=True, quiet=quiet)
    imagej_dir = Path(imagej_dir).expanduser().resolve()
    if not imagej_dir.exists():
        raise FileNotFoundError(f"ImageJ/Fiji folder does not exist: {imagej_dir}")
    psf_creator_dir = Path(psf_creator_dir).expanduser().resolve() if psf_creator_dir else None

    params = _parse_psfgenerator_config(config_path)
    try:
        lateral = float(params["ResLateral"])
        axial = float(params["ResAxial"])
    except KeyError as error:
        raise ValueError(f"Configuration is missing required key {error}: {config_path}") from error

    java_exe = _find_java_exe(imagej_dir)
    psf_jar = _find_psf_jar(imagej_dir, psf_creator_dir)
    classpath = os.pathsep.join([str(imagej_dir / "jars" / "*"), str(psf_jar)])

    with tempfile.TemporaryDirectory(prefix="pft_psf_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        before = set(temporary_path.glob("*.tif*"))
        command = [str(java_exe), "-cp", classpath, "PSFGenerator", str(config_path.resolve())]
        completed = subprocess.run(
            command,
            cwd=temporary_path,
            check=False,
            capture_output=quiet,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr if quiet else "See PSFGenerator console output"
            raise RuntimeError(f"PSFGenerator failed with exit code {completed.returncode}: {message}")
        generated = [path for path in temporary_path.glob("*.tif*") if path not in before]
        if not generated:
            raise FileNotFoundError("PSFGenerator finished without producing a TIFF file")
        newest = max(generated, key=lambda path: path.stat().st_mtime)
        if newest.stat().st_size < expected_min_filesize_bytes:
            raise ValueError(f"Generated PSF appears incomplete: {newest.stat().st_size} bytes")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(newest, out_path)
        _normalize_psf_file_in_place(
            out_path,
            res_lateral_nm=lateral,
            res_axial_nm=axial,
        )
    return out_path


def validate_master_psf_file(
    path: Path,
    *,
    normalize_if_needed: bool = True,
    normalization_atol: float = NORMALIZATION_ATOL,
    res_lateral_nm: float | None = None,
    res_axial_nm: float | None = None,
) -> PSFValidation:
    """Validate one master PSF and normalize it automatically when requested."""
    if not path.is_file():
        return PSFValidation(False, False, 0.0, ("PSF TIFF is missing",))
    issues: list[str] = []
    try:
        array = np.asarray(tiff.imread(str(path)), dtype=np.float32)
    except Exception as error:
        return PSFValidation(False, False, 0.0, (f"PSF TIFF cannot be read: {error}",))
    if array.ndim != 3:
        issues.append(f"PSF must be ZYX, found shape={array.shape}")
    if not np.isfinite(array).all():
        issues.append("PSF contains non-finite values")
    if np.any(array < 0):
        issues.append("PSF contains negative values")
    total = float(np.sum(array, dtype=np.float64)) if array.size else 0.0
    if not np.isfinite(total) or total <= 0:
        issues.append(f"PSF has invalid sum={total}")
    normalized = bool(np.isfinite(total) and abs(total - 1.0) <= normalization_atol)
    if not issues and not normalized and normalize_if_needed:
        _normalize_psf_file_in_place(
            path,
            res_lateral_nm=res_lateral_nm,
            res_axial_nm=res_axial_nm,
        )
        array = np.asarray(tiff.imread(str(path)), dtype=np.float32)
        total = float(array.sum(dtype=np.float64))
        normalized = abs(total - 1.0) <= normalization_atol
    if not normalized:
        issues.append(f"PSF is not normalized: sum={total:.12g}")
    return PSFValidation(not issues, normalized, total, tuple(issues))


def _metadata_path(master_dir: Path) -> Path:
    return master_dir / MASTER_METADATA_BASENAME


def _read_master_metadata(master_dir: Path) -> dict[str, Any]:
    path = _metadata_path(master_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Master PSF metadata is missing: {path}. Run scripts/denoising/create_psf.py first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pft-master-psf-v1":
        raise ValueError(f"Unsupported master PSF metadata schema in {path}")
    return payload


def _value_close(
    first: Any,
    second: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    if first is None or second is None:
        return first is second
    try:
        return bool(np.isclose(float(first), float(second), atol=atol, rtol=rtol))
    except (TypeError, ValueError):
        return first == second


def _compare_voxel_metadata(reference: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for axis in ("z", "y", "x"):
        ref_value = (reference.get("voxel_size_um") or {}).get(axis)
        cur_value = (current.get("voxel_size_um") or {}).get(axis)
        if not _value_close(ref_value, cur_value, atol=1e-7, rtol=1e-5):
            issues.append(f"voxel_size_um.{axis} mismatch: reference={ref_value}, current={cur_value}")
    return issues


def _match_channels_by_wavelength(
    *,
    current_optics: Sequence[ChannelOptics],
    master_channels: Sequence[Mapping[str, Any]],
    tolerance_nm: float,
    master_dir: Path,
) -> tuple[list[MasterPSFMatch], list[str]]:
    matches: list[MasterPSFMatch] = []
    issues: list[str] = []
    used_master_indices: set[int] = set()
    for current in current_optics:
        distances = [
            abs(float(current.wavelength_nm) - float(master["wavelength_nm"]))
            for master in master_channels
        ]
        master_index = int(np.argmin(distances))
        distance = distances[master_index]
        master = master_channels[master_index]
        if distance > tolerance_nm:
            issues.append(
                f"channel {current.index} ({current.name}) wavelength {current.wavelength_nm:g} nm "
                f"does not match a master wavelength within {tolerance_nm:g} nm"
            )
            continue
        if master_index in used_master_indices:
            issues.append(
                f"multiple current channels map to master wavelength {master['wavelength_nm']} nm"
            )
            continue
        used_master_indices.add(master_index)
        matches.append(MasterPSFMatch(
            channel_index=int(current.index),
            channel_name=str(current.name),
            wavelength_nm=float(current.wavelength_nm),
            wavelength_source=str(current.wavelength_source),
            display_color=str(master["display_color"]),
            display_color_hex=str(master["display_color_hex"]),
            psf_path=master_dir / str(master["master_psf_filename"]),
        ))
    if len(used_master_indices) != len(master_channels):
        missing = [
            master_channels[index]["wavelength_nm"]
            for index in range(len(master_channels))
            if index not in used_master_indices
        ]
        issues.append(f"current stack does not provide all master wavelengths; missing={missing}")
    matches.sort(key=lambda item: item.channel_index)
    return matches, issues


def validate_stack_against_master_psfs(
    *,
    project_root: Path,
    zarr_dir: Path,
    level: int = MASTER_LEVEL,
    model: ModelName = "BW",
    channel_wavelength_nm: Mapping[str, float] | None = None,
    wavelength_tolerance_nm: float = DEFAULT_WAVELENGTH_TOLERANCE_NM,
    normalize_psfs: bool = True,
) -> tuple[list[MasterPSFMatch], dict[str, Any]]:
    """Validate a stack and return wavelength-based mappings to master PSFs.

    The function raises ``ValueError`` when the stack is incompatible. It never
    creates stack-specific PSFs.
    """
    if level != MASTER_LEVEL:
        raise ValueError("Master PSFs are defined only for OME-Zarr level 0")
    project_root = project_root.resolve()
    zarr_dir = zarr_dir.resolve()
    master_dir = master_psf_directory(project_root)
    master = _read_master_metadata(master_dir)
    issues: list[str] = []

    if master.get("model") != model:
        issues.append(f"PSF model mismatch: master={master.get('model')}, requested={model}")
    if int(master.get("reference_level", -1)) != MASTER_LEVEL:
        issues.append(f"master metadata has invalid reference level {master.get('reference_level')}")

    current_meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    current_axes = str(current_meta.get("axes") or "").lower()
    if current_axes != str(master.get("reference_axes") or "").lower():
        issues.append(
            f"axes mismatch: reference={master.get('reference_axes')}, current={current_axes}"
        )
    current_shape_zyx = list(_shape_zyx(current_meta))
    if current_shape_zyx != list(master.get("reference_shape_zyx") or []):
        issues.append(
            f"spatial shape mismatch: reference={master.get('reference_shape_zyx')}, "
            f"current={current_shape_zyx}"
        )
    issues.extend(_compare_voxel_metadata(master, current_meta))

    # Optical values are essential for creating the master PSFs, but some source
    # CZI files do not repeat the immersion/sample refractive indices in every
    # converted OME-Zarr. A missing value is therefore treated as unavailable
    # metadata, not as evidence of an acquisition mismatch. In that case the
    # fixed value from the master reference is reused and recorded explicitly.
    # A present value that differs from the master reference still stops the run.
    compatibility_warnings: list[str] = []
    resolved_optical_metadata: dict[str, Any] = {}
    for key in ("objective_na", "refractive_index_immersion", "refractive_index_sample"):
        reference_value = master.get(key)
        current_value = current_meta.get(key)
        if reference_value is None:
            issues.append(
                f"master PSF metadata does not define required optical value {key}"
            )
            continue
        if current_value is None:
            resolved_optical_metadata[key] = reference_value
            compatibility_warnings.append(
                f"{key} is absent from the current OME-Zarr; "
                f"using fixed master-reference value {reference_value}"
            )
            continue
        resolved_optical_metadata[key] = current_value
        if not _value_close(reference_value, current_value, atol=1e-5, rtol=1e-5):
            issues.append(
                f"{key} mismatch: reference={reference_value}, current={current_value}"
            )

    current_optics = resolve_channel_optics(
        zarr_dir,
        level=level,
        explicit_wavelength_nm=channel_wavelength_nm,
    )
    if len(current_optics) != int(master.get("channel_count", -1)):
        issues.append(
            f"channel count mismatch: reference={master.get('channel_count')}, current={len(current_optics)}"
        )
    master_channels = master.get("channels")
    if not isinstance(master_channels, list) or len(master_channels) != 3:
        issues.append("master metadata must contain exactly three channel records")
        master_channels = []

    matches: list[MasterPSFMatch] = []
    if master_channels and len(current_optics) == len(master_channels):
        matches, channel_issues = _match_channels_by_wavelength(
            current_optics=current_optics,
            master_channels=master_channels,
            tolerance_nm=wavelength_tolerance_nm,
            master_dir=master_dir,
        )
        issues.extend(channel_issues)

    for match in matches:
        validation = validate_master_psf_file(
            match.psf_path,
            normalize_if_needed=normalize_psfs,
        )
        if not validation.suitable:
            issues.extend(
                f"{match.psf_path.name}: {issue}" for issue in validation.issues
            )

    if issues:
        formatted = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(
            f"OME-Zarr stack is incompatible with the reusable master PSF set:\n{formatted}"
        )

    # Return a report copy rather than mutating the on-disk master metadata.
    report_metadata = dict(master)
    report_metadata["compatibility_warnings"] = compatibility_warnings
    report_metadata["current_stack_optical_metadata"] = {
        "objective_na": current_meta.get("objective_na"),
        "refractive_index_immersion": current_meta.get("refractive_index_immersion"),
        "refractive_index_sample": current_meta.get("refractive_index_sample"),
    }
    report_metadata["resolved_optical_metadata_for_compatibility"] = resolved_optical_metadata
    report_metadata["missing_current_optics_policy"] = (
        "Use the fixed master-reference value only when the current OME-Zarr value is absent; "
        "stop when a present value differs."
    )
    return matches, report_metadata


def generate_or_reuse_master_psfs(
    *,
    reference_zarr: Path,
    start_path: Path,
    model: ModelName = "BW",
    accuracy: str = "Best",
    quiet: bool = False,
    imagej_dir: Path | None = None,
    psf_creator_dir: Path | None = None,
    channel_wavelength_nm: Mapping[str, float] | None = None,
    reuse_existing: bool = True,
    regenerate: bool = False,
) -> dict[float, Path]:
    """Generate the three level-0 master PSFs once, or reuse a valid set."""
    project_root = find_project_root(start_path.resolve())
    jobs, expected = plan_master_psf_jobs(
        project_root=project_root,
        reference_zarr=reference_zarr,
        model=model,
        channel_wavelength_nm=channel_wavelength_nm,
        accuracy=accuracy,
    )
    master_dir = master_psf_directory(project_root)
    metadata_path = _metadata_path(master_dir)

    existing_metadata: dict[str, Any] | None = None
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "schema", "reference_level", "reference_axes", "reference_shape_zyx",
            "voxel_size_um", "objective_na", "refractive_index_immersion",
            "refractive_index_sample", "channel_count", "model", "accuracy",
        )
        mismatch = [
            key for key in comparable_keys
            if existing_metadata.get(key) != expected.get(key)
        ]
        channel_identity_keys = (
            "reference_channel_index", "reference_channel_name", "wavelength_nm",
            "display_color", "display_color_hex", "master_psf_filename",
        )
        existing_channels = [
            {key: item.get(key) for key in channel_identity_keys}
            for item in (existing_metadata.get("channels") or [])
            if isinstance(item, dict)
        ]
        expected_channels = [
            {key: item.get(key) for key in channel_identity_keys}
            for item in (expected.get("channels") or [])
            if isinstance(item, dict)
        ]
        if existing_channels != expected_channels:
            mismatch.append("channels")
        if mismatch and not regenerate:
            raise ValueError(
                "An incompatible master PSF set already exists. "
                f"Mismatched metadata fields: {mismatch}. Use --regenerate only when the "
                "reference microscope configuration intentionally changed"
            )

    report_lines = [
        "PFT reusable master PSF generation report",
        "=" * 80,
        f"Reference OME-Zarr: {Path(reference_zarr).resolve()}",
        "Reference pyramid level: 0",
        f"PSF model: {model}",
        f"ImageJ/Fiji folder: {imagej_dir or 'project default'}",
        f"PSF Generator folder/JAR: {psf_creator_dir or 'ImageJ plugins/default'}",
        "Master policy: one normalized PSF per wavelength, reused across compatible stacks",
        "",
    ]

    generated_records: list[dict[str, Any]] = []
    outputs: dict[float, Path] = {}
    for job in jobs:
        params = _parse_psfgenerator_config(job.config_txt)
        validation = validate_master_psf_file(
            job.out_tif,
            normalize_if_needed=True,
            res_lateral_nm=_float_or_none(params.get("ResLateral")),
            res_axial_nm=_float_or_none(params.get("ResAxial")),
        )
        can_reuse = bool(
            reuse_existing
            and not regenerate
            and existing_metadata is not None
            and validation.suitable
        )
        if can_reuse:
            action = "REUSED"
        else:
            action = "GENERATED"
            run_psfgenerator_cli_v2(
                start_path=start_path,
                config_path=job.config_txt,
                out_path=job.out_tif,
                imagej_dir=imagej_dir,
                psf_creator_dir=psf_creator_dir,
                quiet=quiet,
            )
            validation = validate_master_psf_file(
                job.out_tif,
                normalize_if_needed=True,
                res_lateral_nm=_float_or_none(params.get("ResLateral")),
                res_axial_nm=_float_or_none(params.get("ResAxial")),
            )
            if not validation.suitable:
                raise ValueError(
                    f"Generated master PSF failed validation: {job.out_tif}\n"
                    + "\n".join(validation.issues)
                )
        array = np.asarray(tiff.imread(str(job.out_tif)), dtype=np.float32)
        generated_records.append({
            "wavelength_nm": float(job.wavelength_nm),
            "wavelength_source": job.wavelength_source,
            "display_color": job.display_color,
            "display_color_hex": job.display_color_hex,
            "reference_channel_index": job.reference_channel_index,
            "reference_channel_name": job.reference_channel_name,
            "master_psf_filename": job.out_tif.name,
            "configuration_filename": str(job.config_txt.relative_to(master_dir)),
            "configuration_sha256": _config_hash(params),
            "psf_shape_zyx": list(array.shape),
            "psf_dtype": str(array.dtype),
            "psf_min": float(array.min()),
            "psf_max": float(array.max()),
            "psf_sum": float(array.sum(dtype=np.float64)),
            "normalized": validation.normalized,
        })
        outputs[float(job.wavelength_nm)] = job.out_tif
        report_lines.extend([
            f"[{action}] {job.display_color.upper()} channel, {job.wavelength_nm:g} nm",
            f"  reference channel: {job.reference_channel_name} (index {job.reference_channel_index})",
            f"  configuration: {job.config_txt}",
            f"  PSF: {job.out_tif}",
            f"  normalized: {validation.normalized}; sum={validation.psf_sum:.12g}",
            "",
        ])

    metadata = dict(expected)
    metadata["channels"] = sorted(generated_records, key=lambda item: item["wavelength_nm"])
    metadata["master_directory"] = str(master_dir.resolve())
    metadata["generated_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["normalization_rule"] = "Every master PSF is non-negative and normalized to sum 1"
    metadata["channel_matching_rule"] = "Match current channels to master PSFs by wavelength metadata"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report_path = master_dir / "psf_master_generation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Final self-test against the reference image verifies the exact code path used later.
    validate_stack_against_master_psfs(
        project_root=project_root,
        zarr_dir=reference_zarr,
        level=MASTER_LEVEL,
        model=model,
        channel_wavelength_nm=channel_wavelength_nm,
        normalize_psfs=True,
    )
    return outputs


def _parse_models_arg(value: str) -> list[ModelName]:
    """Retained parser with a one-model restriction for compatibility."""
    parts = [part.strip().upper() for part in (value or "BW").replace(" ", ",").split(",") if part.strip()]
    output: list[ModelName] = []
    for part in parts:
        if part not in ("BW", "GL", "RW"):
            raise ValueError(f"Invalid PSF model {part!r}; use BW, GL, or RW")
        if part not in output:
            output.append(part)  # type: ignore[arg-type]
    if len(output) != 1:
        raise ValueError("The reusable master workflow supports one active PSF model at a time")
    return output


__all__ = [
    "DEFAULT_WAVELENGTH_TOLERANCE_NM",
    "MASTER_LEVEL",
    "MASTER_METADATA_BASENAME",
    "MasterPSFJob",
    "MasterPSFMatch",
    "ModelName",
    "PSFValidation",
    "REFERENCE_WAVELENGTH_COLORS",
    "STANDARD_CONFIG_BASENAME",
    "_available_levels",
    "_parse_models_arg",
    "_prompt_level",
    "ensure_standard_config",
    "generate_or_reuse_master_psfs",
    "master_psf_directory",
    "plan_master_psf_jobs",
    "run_psfgenerator_cli_v2",
    "update_params_for_image_and_channel",
    "validate_master_psf_file",
    "validate_stack_against_master_psfs",
    "write_psfgenerator_config",
]
