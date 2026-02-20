from __future__ import annotations
from pathlib import Path
import os
import subprocess
import shutil
import tifffile as tiff  
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
    Read the generated PSF TIFF and rewrite it with ImageJ-compatible metadata:
   
    """
    arr = tiff.imread(str(src_tif))  
    resolution = (1.0 / float(res_lateral_nm), 1.0 / float(res_lateral_nm))

    tiff.imwrite(
        str(dst_tif),
        arr.astype("float32", copy=False),
        imagej=True,
        resolution=resolution,
        metadata={
            "unit": unit,
            "spacing": float(res_axial_nm),
        },
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


def main() -> int:
    out = run_psfgenerator_cli(
        start_path=Path(__file__),
        out_name="theoretical_psf.tif",    
        config_name="PSFGenerator.txt",     
        quiet=False,                        
    )
    print(f"Saved PSF: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())