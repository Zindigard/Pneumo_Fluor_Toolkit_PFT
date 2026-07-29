"""
Metadata-driven theoretical PSF generation for 3D OME-Zarr images.

The workflow creates a documented PSFGenerator configuration before every run,
reads channel wavelengths and optical sampling from OME-Zarr metadata, and uses
405/488/561 nm fallbacks only when wavelength metadata are absent. Existing PSFs
are reused only after their array values, normalization, configuration, and
source-image metadata have been validated.

External software requirements
------------------------------
PSF generation requires:

* an ImageJ/Fiji installation containing Java and the ImageJ JAR files;
* a PSF Generator folder or JAR file containing ``PSF_Generator.jar`` or
  ``PSFGenerator.jar``.

Both locations can be provided explicitly by the terminal script. They are
essential when the software is not installed in the project's default Fiji
folder.
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
from typing import Any, Iterable, Literal, Mapping, Optional

import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.metadata_3d import (
    ChannelOptics,
    resolve_channel_optics,
)
from PFT.core_prog_parts.denoising.fuji_managment import (
    ensure_fiji_in_project,
    ensure_psf_generator_exists,
)

ModelName = Literal["BW", "GL", "RW"]
STANDARD_CONFIG_BASENAME = "PSFGenerator_standard"
NORMALIZATION_ATOL = 1e-5


@dataclass(frozen=True)
class PSFJob:
    """One channel-, level-, and model-specific PSF generation task."""

    model: ModelName
    channel_index: int
    channel_name: str
    wavelength_nm: float
    wavelength_source: str
    level: int
    out_tif: Path
    config_txt: Path
    metadata_json: Path


@dataclass(frozen=True)
class PSFValidation:
    """Validation result for an existing or newly generated PSF."""

    suitable: bool
    normalized: bool
    psf_sum: float
    issues: tuple[str, ...]


def _available_levels(zarr_dir: Path) -> list[int]:
    """Return OME-Zarr pyramid-level indices."""
    root = zarr.open_group(str(zarr_dir), mode="r")
    multiscales = root.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return [0]
    datasets = multiscales[0].get("datasets")
    return list(range(len(datasets))) if isinstance(datasets, list) and datasets else [0]


def _prompt_level(levels: list[int], default: int = 2) -> int:
    """Prompt for a valid OME-Zarr pyramid level."""
    if not levels:
        return 0
    selected_default = default if default in levels else levels[0]
    print("\nAvailable OME-Zarr pyramid levels:")
    print("  " + ", ".join(str(level) for level in levels))
    answer = input(f"Choose level [{selected_default}]: ").strip()
    selected = selected_default if answer == "" else int(answer)
    if selected not in levels:
        raise ValueError(f"Invalid level {selected}. Available levels: {levels}")
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
    preferred = ["Type", "Lambda", "NA", "NX", "NY", "NZ", "ResLateral", "ResAxial", "Scale", "unit"]
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
    """Return a conservative standard template used only when no legacy config exists."""
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
    """Create the standard named model configuration before metadata substitution.

    A legacy ``results/psf/config<Model>.txt`` file is used as the source when it
    exists. Otherwise a documented built-in template is created. The resulting
    standard file is always named ``PSFGenerator_standard_<MODEL>.txt``.
    """
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
            "Image-specific metadata are inserted into a separate generated configuration.",
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
    """Insert image sampling, dimensions, optics, and channel wavelength."""
    params = dict(base_params)
    params["Type"] = model
    params["Lambda"] = f"{float(wavelength_nm):.10g}"
    params["unit"] = "nm"

    z_size, y_size, x_size = _shape_zyx(img_meta)
    params["NZ"] = str(z_size)
    # PSFs should remain computationally manageable. Existing standard NX/NY
    # values are retained unless absent, then capped at 256 pixels.
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


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _metadata_signature(
    *,
    zarr_dir: Path,
    img_meta: Mapping[str, Any],
    optics: ChannelOptics,
    model: ModelName,
    level: int,
    params: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "source_omezarr": str(zarr_dir.resolve()),
        "source_level": int(level),
        "source_array_path": img_meta.get("array_path"),
        "source_shape": list(img_meta.get("shape") or []),
        "source_axes": img_meta.get("axes"),
        "voxel_size_um": img_meta.get("voxel_size_um"),
        "objective_na": img_meta.get("objective_na"),
        "refractive_index_immersion": img_meta.get("refractive_index_immersion"),
        "refractive_index_sample": img_meta.get("refractive_index_sample"),
        "channel_index": optics.index,
        "channel_name": optics.name,
        "wavelength_nm": optics.wavelength_nm,
        "wavelength_source": optics.wavelength_source,
        "model": model,
        "config_parameters": dict(params),
    }


def _config_hash(params: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(sorted(params.items())), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_psf_jobs_for_image(
    *,
    project_root: Path,
    zarr_dir: Path,
    level: int,
    models: Iterable[ModelName] = ("BW", "GL", "RW"),
    channel_wavelength_nm: Optional[Mapping[str, float]] = None,
    accuracy: str = "Best",
) -> list[PSFJob]:
    """Create metadata-specific configurations and PSF output paths."""
    project_root = project_root.resolve()
    zarr_dir = zarr_dir.resolve()
    img_meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    optics_list = resolve_channel_optics(
        zarr_dir,
        level=level,
        explicit_wavelength_nm=channel_wavelength_nm,
    )

    generated_dir = project_root / "results" / "psf" / "generated"
    config_dir = project_root / "results" / "psf" / "configuration" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    sample = _safe_name(zarr_dir.parent.name)

    jobs: list[PSFJob] = []
    for model in models:
        standard_config = ensure_standard_config(project_root, model)
        standard_params = _parse_psfgenerator_config(standard_config)
        for optics in optics_list:
            params = update_params_for_image_and_channel(
                standard_params,
                img_meta=img_meta,
                wavelength_nm=optics.wavelength_nm,
                model=model,
                accuracy=accuracy,
            )
            channel = _safe_name(optics.name)
            config_path = config_dir / (
                f"PSFGenerator_config__{sample}__{model}__{channel}__"
                f"Lambda{int(round(optics.wavelength_nm))}nm__L{level}.txt"
            )
            output_path = generated_dir / (
                f"psf_{sample}_{model}_{channel}_"
                f"Lambda{int(round(optics.wavelength_nm))}nm__L{level}.tif"
            )
            metadata_path = output_path.with_suffix(".json")
            signature = _metadata_signature(
                zarr_dir=zarr_dir,
                img_meta=img_meta,
                optics=optics,
                model=model,
                level=level,
                params=params,
            )
            write_psfgenerator_config(
                params,
                config_path,
                header=(
                    "PFT metadata-specific PSFGenerator configuration.",
                    f"Source OME-Zarr: {zarr_dir}",
                    f"Source level: {level}",
                    f"Channel: {optics.name} (index {optics.index})",
                    f"Wavelength: {optics.wavelength_nm:g} nm; source: {optics.wavelength_source}",
                    f"Configuration SHA-256: {_config_hash(params)}",
                    "This file must remain beside the generated PSF audit metadata.",
                ),
            )
            jobs.append(
                PSFJob(
                    model=model,
                    channel_index=optics.index,
                    channel_name=optics.name,
                    wavelength_nm=optics.wavelength_nm,
                    wavelength_source=optics.wavelength_source,
                    level=level,
                    out_tif=output_path,
                    config_txt=config_path,
                    metadata_json=metadata_path,
                )
            )
            # Store expected signature temporarily in the config sidecar, ensuring
            # the configuration document exists before any Java process starts.
            config_path.with_suffix(".expected.json").write_text(
                json.dumps(signature, indent=2, default=str), encoding="utf-8"
            )
    return jobs


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
    if java_home and (Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")).is_file():
        return Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
    system_java = shutil.which("java")
    if system_java:
        return Path(system_java)
    raise FileNotFoundError("Java was not found in ImageJ/Fiji, JAVA_HOME, or PATH.")


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
        "or JAR path containing PSF_Generator.jar."
    )


def _rewrite_with_imagej_metadata(
    src_tif: Path,
    dst_tif: Path,
    *,
    res_lateral_nm: float,
    res_axial_nm: float,
) -> tuple[tuple[int, int, int], float]:
    """Normalize a generated PSF and write calibrated ImageJ-compatible TIFF."""
    array = np.asarray(tiff.imread(str(src_tif)), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D PSF ZYX, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Generated PSF contains NaN or infinite values")
    array = np.maximum(array, 0.0)
    total = float(array.sum(dtype=np.float64))
    if total <= 0:
        raise ValueError("Generated PSF has zero non-negative sum")
    array /= np.float32(total)
    resolution = (1.0 / float(res_lateral_nm), 1.0 / float(res_lateral_nm))
    metadata = {
        "unit": "nm",
        "spacing": float(res_axial_nm),
        "axes": "ZYX",
        "hyperstack": True,
        "mode": "grayscale",
        "channels": 1,
        "slices": int(array.shape[0]),
        "frames": 1,
    }
    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    if dst_tif.exists():
        dst_tif.unlink()
    tiff.imwrite(
        str(dst_tif),
        array,
        imagej=True,
        resolution=resolution,
        metadata=metadata,
        photometric="minisblack",
    )
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
    """Run PSFGenerator and write a normalized calibrated TIFF.

    ``imagej_dir`` and ``psf_creator_dir`` should be passed explicitly on a new
    computer. When omitted, the project Fiji management functions are used.
    """
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
    classpath_items = [str(imagej_dir / "jars" / "*"), str(psf_jar)]
    classpath = os.pathsep.join(classpath_items)

    with tempfile.TemporaryDirectory(prefix="pft_psfgen_") as temporary:
        temporary_dir = Path(temporary)
        command = [str(java_exe), "-cp", classpath, "PSFGenerator", str(config_path.resolve())]
        subprocess.run(
            command,
            cwd=str(temporary_dir),
            check=True,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.STDOUT if quiet else None,
        )
        outputs = list(temporary_dir.glob("*.tif")) + list(temporary_dir.glob("*.tiff"))
        if not outputs:
            raise FileNotFoundError("PSFGenerator finished without producing a TIFF file")
        newest = max(outputs, key=lambda path: path.stat().st_mtime)
        if newest.stat().st_size < expected_min_filesize_bytes:
            raise ValueError(f"Generated PSF appears incomplete: {newest.stat().st_size} bytes")
        _rewrite_with_imagej_metadata(
            newest,
            out_path,
            res_lateral_nm=lateral,
            res_axial_nm=axial,
        )
    return out_path


def _expected_signature_for_job(job: PSFJob) -> dict[str, Any]:
    path = job.config_txt.with_suffix(".expected.json")
    if not path.is_file():
        raise FileNotFoundError(f"Expected PSF signature is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_psf_for_job(job: PSFJob, *, normalization_atol: float = NORMALIZATION_ATOL) -> PSFValidation:
    """Check PSF values and metadata against the source-image-specific job."""
    issues: list[str] = []
    if not job.out_tif.is_file():
        return PSFValidation(False, False, 0.0, ("PSF TIFF is missing",))
    try:
        array = np.asarray(tiff.imread(str(job.out_tif)), dtype=np.float32)
    except Exception as error:
        return PSFValidation(False, False, 0.0, (f"PSF TIFF cannot be read: {error}",))
    if array.ndim != 3:
        issues.append(f"PSF must be ZYX, found shape={array.shape}")
    if not np.isfinite(array).all():
        issues.append("PSF contains non-finite values")
    if np.any(array < 0):
        issues.append("PSF contains negative values")
    total = float(np.sum(array, dtype=np.float64)) if array.size else 0.0
    normalized = bool(np.isfinite(total) and abs(total - 1.0) <= normalization_atol)
    if not normalized:
        issues.append(f"PSF is not normalized: sum={total:.12g}")

    if not job.metadata_json.is_file():
        issues.append("PSF audit metadata JSON is missing")
    else:
        try:
            actual = json.loads(job.metadata_json.read_text(encoding="utf-8"))
            expected = _expected_signature_for_job(job)
            for key in (
                "source_omezarr", "source_level", "source_array_path", "source_shape",
                "source_axes", "voxel_size_um", "objective_na",
                "refractive_index_immersion", "refractive_index_sample",
                "channel_index", "channel_name", "wavelength_nm", "model",
            ):
                if actual.get(key) != expected.get(key):
                    issues.append(f"metadata mismatch for {key}: expected={expected.get(key)!r}, actual={actual.get(key)!r}")
            expected_hash = _config_hash(expected.get("config_parameters") or {})
            if actual.get("config_sha256") != expected_hash:
                issues.append("configuration hash does not match current image-specific configuration")
        except Exception as error:
            issues.append(f"PSF audit metadata cannot be read: {error}")
    return PSFValidation(not issues, normalized, total, tuple(issues))


def _write_psf_audit(job: PSFJob) -> None:
    signature = _expected_signature_for_job(job)
    array = np.asarray(tiff.imread(str(job.out_tif)), dtype=np.float32)
    signature.update({
        "config_file": str(job.config_txt.resolve()),
        "config_sha256": _config_hash(signature.get("config_parameters") or {}),
        "psf_file": str(job.out_tif.resolve()),
        "psf_shape_zyx": list(array.shape),
        "psf_dtype": str(array.dtype),
        "psf_min": float(array.min()),
        "psf_max": float(array.max()),
        "psf_sum": float(array.sum(dtype=np.float64)),
        "normalized": bool(abs(float(array.sum(dtype=np.float64)) - 1.0) <= NORMALIZATION_ATOL),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    })
    job.metadata_json.write_text(json.dumps(signature, indent=2, default=str), encoding="utf-8")


def generate_psfs_for_image(
    *,
    zarr_dir: Path,
    start_path: Path,
    level: int,
    models: Iterable[ModelName],
    accuracy: str = "Best",
    quiet: bool = False,
    imagej_dir: Path | None = None,
    psf_creator_dir: Path | None = None,
    channel_wavelength_nm: Mapping[str, float] | None = None,
    reuse_existing: bool = True,
) -> dict[tuple[str, str, int], Path]:
    """Validate, reuse, or generate all required PSFs for one OME-Zarr image."""
    project_root = find_project_root(start_path.resolve())
    jobs = plan_psf_jobs_for_image(
        project_root=project_root,
        zarr_dir=zarr_dir,
        level=level,
        models=models,
        channel_wavelength_nm=channel_wavelength_nm,
        accuracy=accuracy,
    )
    outputs: dict[tuple[str, str, int], Path] = {}
    report_lines = [
        "PFT metadata-driven PSF generation report",
        "=" * 80,
        f"Source OME-Zarr: {Path(zarr_dir).resolve()}",
        f"Pyramid level: {level}",
        f"ImageJ/Fiji folder: {imagej_dir or 'project default'}",
        f"PSF Generator folder/JAR: {psf_creator_dir or 'ImageJ plugins/default'}",
        "",
    ]
    for job in jobs:
        validation = validate_psf_for_job(job)
        if reuse_existing and validation.suitable:
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
            _write_psf_audit(job)
            validation = validate_psf_for_job(job)
            if not validation.suitable:
                raise ValueError(
                    f"Generated PSF failed validation: {job.out_tif}\n" + "\n".join(validation.issues)
                )
        outputs[(job.model, job.channel_name, job.level)] = job.out_tif
        report_lines.extend([
            f"[{action}] model={job.model} channel={job.channel_name} level={job.level}",
            f"  wavelength_nm={job.wavelength_nm:g} ({job.wavelength_source})",
            f"  config={job.config_txt}",
            f"  psf={job.out_tif}",
            f"  normalized={validation.normalized} sum={validation.psf_sum:.12g}",
            "  suitability=PASS" if validation.suitable else "  suitability=FAIL",
            *(f"  issue: {issue}" for issue in validation.issues),
            "",
        ])
    sample = _safe_name(Path(zarr_dir).parent.name)
    report_path = (
        project_root / "results" / "psf" /
        f"psf_generation_report__{sample}__L{level}.txt"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return outputs


def _parse_models_arg(value: str) -> list[ModelName]:
    parts = [part.strip().upper() for part in (value or "BW,GL,RW").replace(" ", ",").split(",") if part.strip()]
    output: list[ModelName] = []
    for part in parts:
        if part not in ("BW", "GL", "RW"):
            raise ValueError(f"Invalid PSF model {part!r}; use BW, GL, or RW")
        if part not in output:
            output.append(part)  # type: ignore[arg-type]
    return output


__all__ = [
    "ModelName",
    "PSFJob",
    "PSFValidation",
    "STANDARD_CONFIG_BASENAME",
    "_available_levels",
    "_prompt_level",
    "ensure_standard_config",
    "generate_psfs_for_image",
    "plan_psf_jobs_for_image",
    "run_psfgenerator_cli_v2",
    "update_params_for_image_and_channel",
    "validate_psf_for_job",
    "write_psfgenerator_config",
]
