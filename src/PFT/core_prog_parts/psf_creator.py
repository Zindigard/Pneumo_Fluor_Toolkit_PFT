from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional
import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.fuji_managment import (
    find_project_root,
    ensure_fiji_in_project,
    ensure_psf_generator_exists,
)
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare

"Script to generate theoretical PSFs using PSFGenerator for OME-Zarr images."


def _available_levels(zarr_dir: Path) -> list[int]:
    """
    Read OME-NGFF multiscales datasets list and return indices [0..L-1].
    If multiscales is missing, fall back to [0].
    """
    root = zarr.open_group(str(zarr_dir), mode="r")
    ms = root.attrs.get("multiscales")
    if not ms or not isinstance(ms, list) or not ms[0].get("datasets"):
        return [0]
    ds = ms[0]["datasets"]
    return list(range(len(ds)))


def _prompt_level(levels: list[int], default: int = 2) -> int:
    """Internal helper used by this module."""
    if not levels:
        return 0
    print("\nAvailable OME-Zarr pyramid levels:")
    print("  " + ", ".join(str(i) for i in levels))
    s = input(f"Choose level [{default}]: ").strip()
    chosen = default if s == "" else int(s)
    if chosen not in levels:
        raise SystemExit(f"Invalid level {chosen}. Must be one of: {levels}")
    return chosen


def _find_java_exe(fiji_dir: Path) -> Path:
    """Internal helper used by this module."""
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


def write_psfgenerator_config(params: Mapping[str, str], dst: Path) -> None:
    """Write the requested report or metadata file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={params[k]}" for k in sorted(params.keys())]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rewrite_with_imagej_metadata(
    src_tif: Path,
    dst_tif: Path,
    *,
    res_lateral_nm: float,
    res_axial_nm: float,
    unit: str = "nm",
) -> None:
    """Internal helper used by this module."""
    arr = tiff.imread(str(src_tif)).astype("float32", copy=False)

    arr = np.maximum(arr, 0)
    s = float(arr.sum())
    if s <= 0:
        raise ValueError(f"PSF sum is zero after clipping negatives: {src_tif}")
    arr /= s

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (Z,Y,X). Got shape={arr.shape}")

    z, _, _ = arr.shape

    pixels_per_nm = 1.0 / float(res_lateral_nm)
    resolution = (pixels_per_nm, pixels_per_nm)

    ij_meta = {
        "unit": unit,
        "spacing": float(res_axial_nm),  # Z-step
        "axes": "ZYX",
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
        photometric="minisblack",
    )


def run_psfgenerator_cli_v2(
    *,
    start_path: Path,
    config_path: Path,
    out_path: Path,
    quiet: bool = False,
    expected_min_filesize_bytes: int = 1024,
) -> Path:
    """
    Run PSFGenerator (Java) for a given config, then rewrite output to `out_path`
    with safe normalization + ImageJ metadata.
    """
    project_root = find_project_root(start_path.resolve())

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

    java_exe = _find_java_exe(fiji_dir)

    jar_a = fiji_dir / "plugins" / "PSF_Generator.jar"
    jar_b = fiji_dir / "plugins" / "PSFGenerator.jar"
    jar_plugin = jar_a if jar_a.exists() else jar_b if jar_b.exists() else None
    if jar_plugin is None:
        raise FileNotFoundError(f"PSF Generator jar not found in: {fiji_dir / 'plugins'}")

    classpath = f"{(fiji_dir / 'jars' / '*')};{jar_plugin}"

    with tempfile.TemporaryDirectory(prefix="pft_psfgen_") as td:
        td = Path(td)

        before = {p.name: p.stat().st_mtime for p in (list(td.glob("*.tif")) + list(td.glob("*.tiff")))}

        cmd = [str(java_exe), "-cp", classpath, "PSFGenerator", str(config_path)]
        subprocess.run(cmd, cwd=str(td), check=True)

        after = list(td.glob("*.tif")) + list(td.glob("*.tiff"))
        changed: list[Path] = []
        for p in after:
            try:
                mtime = p.stat().st_mtime
                if (p.name not in before) or (mtime > before[p.name] + 1e-6):
                    changed.append(p)
            except FileNotFoundError:
                continue

        if not changed:
            raise FileNotFoundError(f"No new/updated TIFF created by PSFGenerator in temp dir: {td}")

        newest = max(changed, key=lambda p: p.stat().st_mtime)
        if newest.stat().st_size < expected_min_filesize_bytes:
            raise FileNotFoundError(
                f"PSFGenerator created '{newest.name}', but it looks too small ({newest.stat().st_size} bytes)."
            )

        if out_path.exists():
            out_path.unlink()
        _rewrite_with_imagej_metadata(
            newest,
            out_path,
            res_lateral_nm=res_lateral_nm,
            res_axial_nm=res_axial_nm,
            unit=unit,
        )

    return out_path


ModelName = Literal["BW", "GL", "RW"]


@dataclass(frozen=True)
class PSFJob:
    model: ModelName
    channel_name: str
    wavelength_nm: float
    level: int
    out_tif: Path
    config_txt: Path


def _default_channel_wavelengths_from_names(channel_names: list[str]) -> dict[str, float]:
    """
    Your common convention:
      T1 -> 405 nm, T2 -> 488 nm, T3 -> 561 nm
    """
    out: dict[str, float] = {}
    for ch in channel_names:
        up = ch.upper()
        if "T1" in up:
            out[ch] = 405.0
        elif "T2" in up:
            out[ch] = 488.0
        elif "T3" in up:
            out[ch] = 561.0
    return out


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
  
    """Update parameters or metadata in place and return results."""
    p = dict(base_params)

    p["Lambda"] = str(float(wavelength_nm))
    p["unit"] = "nm"
    p["NZ"] = p.get("NZ", "40")  

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


def plan_psf_jobs_for_image(
    *,
    project_root: Path,
    zarr_dir: Path,
    level: int,
    models: Iterable[ModelName] = ("BW", "GL", "RW"),
    channel_wavelength_nm: Optional[Mapping[str, float]] = None,
    accuracy: str = "Best",
) -> list[PSFJob]:
   
    """Prepare the list of jobs to run next."""
    project_root = project_root.resolve()
    zarr_dir = zarr_dir.resolve()

    img_meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)

    channel_names = img_meta.get("channel_names") or []
    if not isinstance(channel_names, list) or len(channel_names) == 0:
        raise ValueError(f"No channel_names found in OME-Zarr attrs for: {zarr_dir}")

    if channel_wavelength_nm is None:
        channel_wavelength_nm = _default_channel_wavelengths_from_names(list(channel_names))

    cfg_dir = project_root / "results" / "psf"
    gen_dir = cfg_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[PSFJob] = []
    for model in models:
        base_cfg = cfg_dir / f"config{model}.txt"
        if not base_cfg.exists():
            raise FileNotFoundError(
                f"Base config for model {model} not found: {base_cfg}\n"
                f"Expected files like: configBW.txt, configGL.txt, configRW.txt in {cfg_dir}"
            )
        base_params = _parse_psfgenerator_config(base_cfg)

        for ch in channel_names:
            if ch not in channel_wavelength_nm:
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

            cfg_name = f"PSFGenerator_{model}_{ch}_Lambda{int(round(lam))}nm__L{level}.txt"
            out_name = f"psf_{model}_{ch}_Lambda{int(round(lam))}nm__L{level}.tif"

            cfg_path = gen_dir / cfg_name
            out_tif = gen_dir / out_name

            write_psfgenerator_config(params, cfg_path)
            jobs.append(PSFJob(model=model, channel_name=ch, wavelength_nm=lam, level=level, out_tif=out_tif, config_txt=cfg_path))

    if not jobs:
        raise ValueError(
            "No PSF jobs planned. Your channel names may not include T1/T2/T3.\n"
            "If so, provide an explicit wavelength mapping in code, or rename channels consistently."
        )
    return jobs


def generate_psfs_for_image(
    *,
    zarr_dir: Path,
    start_path: Path,
    level: int,
    models: Iterable[ModelName],
    accuracy: str = "Best",
    quiet: bool = False,
) -> dict[tuple[str, str, int], Path]:
    """
    Generate PSFs for one OME-Zarr at a chosen pyramid level.
    Returns dict[(model, channel_name, level)] = psf_path
    """
    project_root = find_project_root(start_path.resolve())
    jobs = plan_psf_jobs_for_image(
        project_root=project_root,
        zarr_dir=zarr_dir,
        level=level,
        models=models,
        accuracy=accuracy,
    )

    outputs: dict[tuple[str, str, int], Path] = {}
    for job in jobs:
        out = run_psfgenerator_cli_v2(
            start_path=start_path,
            config_path=job.config_txt,
            out_path=job.out_tif,
            quiet=quiet,
        )
        outputs[(job.model, job.channel_name, job.level)] = out
    return outputs


def _parse_models_arg(s: str) -> list[ModelName]:
    """Internal helper used by this module."""
    s = (s or "").strip()
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    if not parts:
        return ["BW", "GL", "RW"]
    out: list[ModelName] = []
    for p in parts:
        if p not in ("BW", "GL", "RW"):
            raise ValueError(f"Invalid model '{p}'. Use BW,GL,RW.")
        out.append(p)  
    return out


def main() -> int:
    """Helper function used by this module."""
    ap = argparse.ArgumentParser(
        description="Generate theoretical PSFs for an OME-Zarr, matching a chosen pyramid level."
    )
    ap.add_argument("--zarr_dir", required=True, help="Path to image.ome.zarr")
    ap.add_argument(
        "--level",
        type=int,
        default=None,
        help="OME-Zarr pyramid level. If omitted, script will prompt you interactively.",
    )
    ap.add_argument(
        "--models",
        default="BW,GL,RW",
        help="Comma-separated models. Example: 'BW' or 'BW,GL'. Default: BW,GL,RW",
    )
    ap.add_argument("--accuracy", default="Best", help="PSFGenerator accuracy value (default: Best)")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    zarr_dir = Path(args.zarr_dir)
    if not zarr_dir.exists():
        raise SystemExit(f"Missing: {zarr_dir}")

    if args.level is None:
        levels = _available_levels(zarr_dir)
        level = _prompt_level(levels, default=2)
    else:
        level = int(args.level)

    models = _parse_models_arg(args.models)
    start_path = Path(__file__).resolve()

    print(f"\nPSF generation settings:")
    print(f"  zarr_dir : {zarr_dir}")
    print(f"  level    : {level}")
    print(f"  models   : {models}")
    print(f"  accuracy : {args.accuracy}")

    outs = generate_psfs_for_image(
        zarr_dir=zarr_dir,
        start_path=start_path,
        level=level,
        models=models,
        accuracy=str(args.accuracy),
        quiet=bool(args.quiet),
    )

    print("\nGenerated PSFs:")
    for (model, ch, lvl), p in sorted(outs.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        print(f"  L{lvl}  {model:<2}  {ch:<20}  ->  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())