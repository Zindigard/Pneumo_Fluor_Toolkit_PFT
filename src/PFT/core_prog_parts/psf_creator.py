from __future__ import annotations
from pathlib import Path
import os
import subprocess
import shutil
import tifffile as tiff
import numpy as np
from dataclasses import dataclass
from typing import Mapping, Iterable, Optional, Literal

try:
    from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
except Exception:
    extract_ome_zarr_meta_for_compare = None  # type: ignore

from PFT.core_prog_parts.fuji_managment import (
    find_project_root,
    ensure_fiji_in_project,
    ensure_psf_generator_exists,
)


def _find_java_exe(fiji_dir: Path) -> Path:
    candidates = [
        fiji_dir / "java" / "win64" / "bin" / "java.exe",
        fiji_dir / "java" / "bin" / "java.exe",
        fiji_dir / "jre" / "bin" / "java.exe",
    ]
    for c in candidates:
        if c.exists():
            return c

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        c = Path(java_home) / "bin" / "java.exe"
        if c.exists():
            return c

    where = shutil.which("java")
    if where:
        return Path(where)

    raise FileNotFoundError(
        "Java not found. Install Java (JRE/JDK) or set JAVA_HOME, or use a Fiji build that bundles Java."
    )


def _parse_psfgenerator_config(cfg_path: Path) -> dict[str, str]:
    """
    Parse PSFGenerator.txt style key=value file.
    Keeps only simple key=value lines (skips comments and blanks).
    """
    params: dict[str, str] = {}
    for line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        params[k.strip()] = v.strip()
    return params


def _rewrite_with_imagej_metadata(
    src_tif: Path,
    dst_tif: Path,
    *,
    res_lateral_nm: float,
    res_axial_nm: float,
    unit: str = "nm",
) -> None:
    """
    Read the generated PSF TIFF and rewrite it with ImageJ-compatible metadata.
    """
    arr = tiff.imread(str(src_tif)).astype("float32", copy=False)

    # --- PSF normalization for deconvolution ---
    # 1) enforce non-negativity (guards against tiny numerical negatives)
    # 2) normalize so that sum(PSF) == 1 (energy conservation)
    arr = np.maximum(arr, 0)
    s = float(arr.sum())
    if s > 0:
        arr = arr / s
    else:
        raise ValueError(f"PSF sum is zero after clipping negatives: {src_tif}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (Z,Y,X). Got shape={arr.shape}")

    z, y, x = arr.shape

    pixels_per_nm = 1.0 / float(res_lateral_nm)
    resolution = (pixels_per_nm, pixels_per_nm)

    ij_meta = {
        "unit": unit,                  # "nm"
        "spacing": float(res_axial_nm), # Z-step in nm
        "axes": "ZYX",                  # helps some readers
        "hyperstack": True,
        "mode": "grayscale",
        "channels": 1,
        "slices": int(z),
        "frames": 1,
    }

    tiff.imwrite(
        str(dst_tif),
        arr,
        imagej=True,
        resolution=resolution,
        metadata=ij_meta,
    )


def run_psfgenerator_cli(
    *,
    start_path: Path,
    out_name: str = "theoretical_psf.tif",
    config_name: str = "PSFGenerator.txt",
    quiet: bool = False,
    expected_min_filesize_bytes: int = 1024,
) -> Path:
    """
    Output is written in the same folder and normalized/rewritten to <out_name>
    WITH ImageJ metadata (unit + spacing + XY resolution) so Fiji reads calibration.
    """
    project_root = find_project_root(start_path.resolve())

    # Ensure Fiji + plugin exist (downloads if needed)
    fiji_dir = ensure_fiji_in_project(project_root, quiet=quiet)
    ensure_psf_generator_exists(fiji_dir, auto_download=True, quiet=quiet)

    out_dir = project_root / "results" / "psf"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    params = _parse_psfgenerator_config(config_path)
    try:
        res_lateral_nm = float(params["ResLateral"])  # XY nm per pixel
        res_axial_nm = float(params["ResAxial"])      # Z nm per slice
    except KeyError as e:
        raise FileNotFoundError(
            f"Missing required key {e} in {config_path}. "
            "Expected at least ResLateral and ResAxial."
        )
    unit = params.get("unit", "nm")  # PSFGenerator.txt may not contain it; default nm

    # Find Java
    java_exe = _find_java_exe(fiji_dir)

    # Plugin jar
    jar_a = fiji_dir / "plugins" / "PSF_Generator.jar"
    jar_b = fiji_dir / "plugins" / "PSFGenerator.jar"
    jar_plugin = jar_a if jar_a.exists() else jar_b if jar_b.exists() else None
    if jar_plugin is None:
        raise FileNotFoundError(f"PSF Generator jar not found in: {fiji_dir / 'plugins'}")

    classpath = f"{(fiji_dir / 'jars' / '*')};{jar_plugin}"

    final_out = out_dir / out_name
    if final_out.exists():
        final_out.unlink()

    def list_tifs() -> list[Path]:
        return list(out_dir.glob("*.tif")) + list(out_dir.glob("*.tiff"))

    before = {p.name: p.stat().st_mtime for p in list_tifs()}

    cmd = [str(java_exe), "-cp", classpath, "PSFGenerator", str(config_path)]
    subprocess.run(cmd, cwd=str(out_dir), check=True)

    after = list_tifs()
    changed: list[Path] = []
    for p in after:
        try:
            mtime = p.stat().st_mtime
            if (p.name not in before) or (mtime > before[p.name] + 1e-6):
                changed.append(p)
        except FileNotFoundError:
            continue

    if not changed:
        raise FileNotFoundError(
            f"No new/updated TIFF created in {out_dir}. "
            f"Existing TIFFs before run: {sorted(before.keys())}"
        )

    newest = max(changed, key=lambda p: p.stat().st_mtime)

    if newest.stat().st_size < expected_min_filesize_bytes:
        raise FileNotFoundError(
            f"PSFGenerator created/updated '{newest.name}', but it looks too small "
            f"({newest.stat().st_size} bytes)."
        )

    tmp_src = newest
    _rewrite_with_imagej_metadata(
        tmp_src,
        final_out,
        res_lateral_nm=res_lateral_nm,
        res_axial_nm=res_axial_nm,
        unit=unit,
    )

    orig_copy = out_dir / f"{tmp_src.stem}_orig{tmp_src.suffix}"
    try:
        shutil.move(str(tmp_src), str(orig_copy))
    except Exception:
        pass

    return final_out


ModelName = Literal["BW", "GL", "RW"]


@dataclass(frozen=True)
class PSFJob:
    model: ModelName
    channel_name: str
    wavelength_nm: float
    out_tif: Path
    config_txt: Path


def write_psfgenerator_config(params: Mapping[str, str], dst: Path) -> None:
    """
    Write PSFGenerator config as key=value text file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in sorted(params.keys()):
        v = params[k]
        lines.append(f"{k}={v}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    """
    Return a modified copy of base PSFGenerator parameters for:
    - given PSF model (BW/GL/RW)
    - given channel wavelength (Lambda)
    - optionally match NA and sampling (ResLateral/ResAxial) to image voxel size
    - optionally match refractive indices to image metadata
    """
    p = dict(base_params)

    p["Lambda"] = str(float(wavelength_nm))

    p["unit"] = "nm"
    p["NZ"] = "40"

    if match_na_to_image:
        na = img_meta.get("objective_na")
        if na is not None:
            p["NA"] = str(float(na))

    if match_sampling_to_image:
        vox = img_meta.get("voxel_size_um") or {}
        vx = vox.get("x")
        vz = vox.get("z")
        if vx is not None:
            p["ResLateral"] = str(float(vx) * 1000.0)
        if vz is not None:
            p["ResAxial"] = str(float(vz) * 1000.0)

    if match_refractive_indices:
        ri_imm = img_meta.get("refractive_index_immersion")
        ri_sam = img_meta.get("refractive_index_sample")

        if ri_imm is not None:
            p[f"psf-{model}-NI"] = str(float(ri_imm))
        if model == "GL" and ri_sam is not None:
            p["psf-GL-NS"] = str(float(ri_sam))

    p[f"psf-{model}-accuracy"] = str(accuracy)

    return p


def _default_channel_wavelengths_from_names(channel_names: list[str]) -> dict[str, float]:
    """
    Default SIM channel mapping:

      - TV1-T1-SR  -> 405 nm
      - TV1-T2-SR  -> 488 nm
      - TV1-T3-SR  -> 561 nm
    """
    out: dict[str, float] = {}
    for ch in channel_names:
        if "T1" in ch:
            out[ch] = 405.0
        elif "T2" in ch:
            out[ch] = 488.0
        elif "T3" in ch:
            out[ch] = 561.0
    return out


def plan_psf_jobs_for_image(
    *,
    project_root: Path,
    zarr_dir: Path,
    models: Iterable[ModelName] = ("BW", "GL", "RW"),
    channel_wavelength_nm: Optional[Mapping[str, float]] = None,
    out_dir: Optional[Path] = None,
    accuracy: str = "Best",
) -> list[PSFJob]:
    """
    Create PSF generation jobs (one per model per channel).

    Expects PSFs written into the same folder (by run_psfgenerator_cli).
    """
    if extract_ome_zarr_meta_for_compare is None:
        raise ImportError("extract_ome_zarr_meta_for_compare not available. Ensure decoder_omezar.py is importable.")

    project_root = project_root.resolve()
    zarr_dir = zarr_dir.resolve()
    cfg_dir = project_root / "results" / "psf"
    gen_dir = cfg_dir / "generated"
    out_dir = out_dir or gen_dir

    img_meta = extract_ome_zarr_meta_for_compare(zarr_dir)
    channel_names = img_meta.get("channel_names") or []
    if not channel_names:
        raise ValueError(f"No channel_names found in OME-Zarr attrs for: {zarr_dir}")

    if channel_wavelength_nm is None:
        channel_wavelength_nm = _default_channel_wavelengths_from_names(list(channel_names))

    jobs: list[PSFJob] = []
    for model in models:
        base_cfg = cfg_dir / f"config{model}.txt"
        if not base_cfg.exists():
            raise FileNotFoundError(f"Base config for model {model} not found: {base_cfg}")
        base_params = _parse_psfgenerator_config(base_cfg)

        for ch in channel_names:
            if ch not in channel_wavelength_nm:
                # Skip silently; user can provide mapping
                continue
            lam = float(channel_wavelength_nm[ch])

            params = update_params_for_image_and_channel(
                base_params,
                img_meta=img_meta,
                wavelength_nm=lam,
                model=model,
                accuracy=accuracy,
                match_sampling_to_image=True,
                match_na_to_image=True,
                match_refractive_indices=True,
            )

            cfg_name = f"PSFGenerator_{model}_{ch}_Lambda{int(round(lam))}nm.txt"
            out_name = f"psf_{model}_{ch}_Lambda{int(round(lam))}nm.tif"
            cfg_path = gen_dir / cfg_name
            out_tif = out_dir / out_name

            jobs.append(PSFJob(model=model, channel_name=ch, wavelength_nm=lam, out_tif=out_tif, config_txt=cfg_path))

            write_psfgenerator_config(params, cfg_path)

    if not jobs:
        raise ValueError(
            "No PSF jobs planned. Either channel_names missing, or channel_wavelength_nm mapping did not match channels."
        )
    return jobs


def run_psfgenerator_cli_v2(
    *,
    start_path: Path,
    config_path: Path,
    out_path: Path,
    quiet: bool = False,
    expected_min_filesize_bytes: int = 1024,
) -> Path:
    """
    V2 runner: same logic as run_psfgenerator_cli, but takes explicit config_path and output path.
    Output is rewritten WITH ImageJ metadata (unit + spacing + XY resolution) so Fiji reads calibration.
    """
    project_root = find_project_root(start_path.resolve())

    # Ensure Fiji + plugin exist
    fiji_dir = ensure_fiji_in_project(project_root, quiet=quiet)
    ensure_psf_generator_exists(fiji_dir, auto_download=True, quiet=quiet)

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    params = _parse_psfgenerator_config(config_path)
    try:
        res_lateral_nm = float(params["ResLateral"])
        res_axial_nm = float(params["ResAxial"])
    except KeyError as e:
        raise FileNotFoundError(
            f"Missing required key {e} in {config_path}. Expected at least ResLateral and ResAxial."
        )
    unit = params.get("unit", "nm")

    # Find Java
    java_exe = _find_java_exe(fiji_dir)

    # Plugin jar
    jar_a = fiji_dir / "plugins" / "PSF_Generator.jar"
    jar_b = fiji_dir / "plugins" / "PSFGenerator.jar"
    jar_plugin = jar_a if jar_a.exists() else jar_b if jar_b.exists() else None
    if jar_plugin is None:
        raise FileNotFoundError(f"PSF Generator jar not found in: {fiji_dir / 'plugins'}")

    classpath = f"{(fiji_dir / 'jars' / '*')};{jar_plugin}"

    if out_path.exists():
        out_path.unlink()

    def list_tifs() -> list[Path]:
        return list(out_dir.glob("*.tif")) + list(out_dir.glob("*.tiff"))

    before = {p.name: p.stat().st_mtime for p in list_tifs()}

    cmd = [str(java_exe), "-cp", classpath, "PSFGenerator", str(config_path)]
    subprocess.run(cmd, cwd=str(out_dir), check=True)

    after = list_tifs()
    changed: list[Path] = []
    for p in after:
        try:
            mtime = p.stat().st_mtime
            if (p.name not in before) or (mtime > before[p.name] + 1e-6):
                changed.append(p)
        except FileNotFoundError:
            continue

    if not changed:
        raise FileNotFoundError(
            f"No new/updated TIFF created in {out_dir}. Existing TIFFs before run: {sorted(before.keys())}"
        )

    newest = max(changed, key=lambda p: p.stat().st_mtime)

    if newest.stat().st_size < expected_min_filesize_bytes:
        raise FileNotFoundError(
            f"PSFGenerator created/updated '{newest.name}', but it looks too small ({newest.stat().st_size} bytes)."
        )

    tmp_src = newest
    _rewrite_with_imagej_metadata(
        tmp_src,
        out_path,
        res_lateral_nm=res_lateral_nm,
        res_axial_nm=res_axial_nm,
        unit=unit,
    )

    # Keep original produced file too
    orig_copy = out_dir / f"{tmp_src.stem}_orig{tmp_src.suffix}"
    try:
        if tmp_src.resolve() != out_path.resolve():
            shutil.move(str(tmp_src), str(orig_copy))
    except Exception:
        pass

    return out_path


def generate_psfs_for_image_all_models(
    *,
    zarr_dir: Path,
    start_path: Path,
    channel_wavelength_nm: Optional[Mapping[str, float]] = None,
    models: Iterable[ModelName] = ("BW", "GL", "RW"),
    accuracy: str = "Best",
    quiet: bool = False,
) -> dict[tuple[str, str], Path]:
    """
    For one OME-Zarr image, generate PSFs for each (model x channel).

    Returns
    -------
    dict[(model, channel)] = path_to_psf_tif
    """
    project_root = find_project_root(start_path.resolve())
    jobs = plan_psf_jobs_for_image(
        project_root=project_root,
        zarr_dir=zarr_dir,
        models=models,
        channel_wavelength_nm=channel_wavelength_nm,
        accuracy=accuracy,
    )

    outputs: dict[tuple[str, str], Path] = {}
    for job in jobs:
        out = run_psfgenerator_cli_v2(
            start_path=start_path,
            config_path=job.config_txt,
            out_path=job.out_tif,
            quiet=quiet,
        )
        outputs[(job.model, job.channel_name)] = out
    return outputs