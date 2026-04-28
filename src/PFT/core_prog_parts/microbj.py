from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


"""
Core logic for running MicrobeJ on TIFF images from Python.

"""


@dataclass(frozen=True)
class MicrobeJRunResult:
    image_path: Path
    output_dir: Path
    experiment_name: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def find_project_root(start: Path | None = None) -> Path:
    """
    Lightweight fallback project-root finder.

    If your existing project already has PFT.core_prog_parts.fuji_managment.find_project_root,
    this function will try to use it first. Otherwise it walks upward until it finds a folder
    that looks like Pneumo_Fluor_Toolkit_PFT.
    """
    try:
        from PFT.core_prog_parts.fuji_managment import find_project_root as _find_project_root

        return Path(_find_project_root((start or Path(__file__)).resolve()))
    except Exception:
        pass

    p = (start or Path(__file__)).resolve()
    if p.is_file():
        p = p.parent

    markers = {"src", "scripts", "results"}
    for parent in [p, *p.parents]:
        if parent.name == "Pneumo_Fluor_Toolkit_PFT":
            return parent
        if markers.issubset({x.name for x in parent.iterdir()}):
            return parent

    raise FileNotFoundError(
        "Could not find project root. Start this script from inside Pneumo_Fluor_Toolkit_PFT."
    )


def ensure_fiji_dir(project_root: Path) -> Path:
    """
    Find Fiji in the project using your existing helper when available.
    Expected common location:
        <project_root>/Fiji.app
    """
    try:
        from PFT.core_prog_parts.fuji_managment import ensure_fiji_in_project

        return Path(ensure_fiji_in_project(project_root, quiet=True))
    except Exception:
        pass

    candidates = [
        project_root / "Fiji.app",
        project_root / "fiji" / "Fiji.app",
        project_root / "tools" / "Fiji.app",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Fiji.app was not found. Put Fiji.app inside the project root, or make sure "
        "PFT.core_prog_parts.fuji_managment.ensure_fiji_in_project works."
    )


def find_fiji_executable(fiji_dir: Path) -> Path:
    """Find Fiji/ImageJ executable on Windows/Linux/macOS."""
    candidates = [
        fiji_dir / "ImageJ-win64.exe",
        fiji_dir / "ImageJ-win32.exe",
        fiji_dir / "Fiji.exe",
        fiji_dir / "ImageJ-linux64",
        fiji_dir / "ImageJ-linux32",
        fiji_dir / "ImageJ-macosx",
        fiji_dir / "Contents" / "MacOS" / "ImageJ-macosx",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No Fiji/ImageJ executable found in: {fiji_dir}")


def find_microbej_jar(project_root: Path, explicit_jar: Path | None = None) -> Path:
    """
    Locate MicrobeJ_.jar.

    Best project location:
        <project_root>/tools/microbej/MicrobeJ_.jar

    It can also already be in Fiji.app/plugins.
    """
    if explicit_jar is not None:
        explicit_jar = Path(explicit_jar)
        if explicit_jar.exists():
            return explicit_jar
        raise FileNotFoundError(f"MicrobeJ jar not found: {explicit_jar}")

    candidates = [
        project_root / "tools" / "microbej" / "MicrobeJ_.jar",
        project_root / "resources" / "java" / "MicrobeJ_.jar",
        project_root / "Fiji.app" / "plugins" / "MicrobeJ_.jar",
        project_root / "fiji" / "Fiji.app" / "plugins" / "MicrobeJ_.jar",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(project_root.glob("**/MicrobeJ*.jar"))
    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(
        "MicrobeJ_.jar not found. Recommended: put it here:\n"
        f"  {project_root / 'tools' / 'microbej' / 'MicrobeJ_.jar'}"
    )


def install_microbej_into_fiji(
    *,
    project_root: Path,
    fiji_dir: Path,
    microbej_jar: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Copy MicrobeJ_.jar into Fiji.app/plugins so Fiji can see the plugin.
    """
    src = find_microbej_jar(project_root, microbej_jar)
    plugins_dir = fiji_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    dst = plugins_dir / "MicrobeJ_.jar"
    if dst.exists() and not overwrite:
        return dst

    shutil.copy2(src, dst)
    return dst


def _safe_macro_arg_value(path_or_text: str | Path) -> str:
    """
    MicrobeJ_m internally splits options by spaces, so paths with spaces can break.
    This function fails early with a clear message instead of silently producing wrong results.
    """
    s = str(path_or_text)
    if " " in s:
        raise ValueError(
            "MicrobeJ_m splits macro options by spaces. Please use paths without spaces:\n"
            f"  {s}"
        )
    return s.replace("\\", "/")


def make_microbej_macro(
    *,
    image_path: Path,
    settings_path: Path | None,
    output_dir: Path,
    experiment_name: str,
    debug: bool = False,
) -> str:
    """Create the ImageJ macro text that calls MicrobeJ_m once."""
    image = _safe_macro_arg_value(image_path)
    output = _safe_macro_arg_value(output_dir)
    name = _safe_macro_arg_value(experiment_name)

    parts = [f"image={image}", f"output={output}", f"name={name}", f"debug={'on' if debug else 'off'}"]
    if settings_path is not None:
        parts.insert(0, f"settings={_safe_macro_arg_value(settings_path)}")

    options = " ".join(parts)
    return f'run("MicrobeJ_m", "{options}");\n'


def run_microbej_on_tiff(
    *,
    image_path: Path,
    output_dir: Path,
    settings_path: Path | None,
    project_root: Path | None = None,
    microbej_jar: Path | None = None,
    experiment_name: str | None = None,
    debug: bool = False,
    timeout_s: int | None = None,
    overwrite_plugin: bool = False,
) -> MicrobeJRunResult:
    """
    Run MicrobeJ_m on one TIFF image.

    Parameters
    ----------
    image_path:
        Input .tif/.tiff image.
    output_dir:
        Folder where MicrobeJ should save CSV/template outputs.
    settings_path:
        MicrobeJ .xml or .ini settings file. You should export/save this from MicrobeJ GUI first.
        If None, MicrobeJ tries to use its default internal INI path.
    """
    t0 = time.time()
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    settings_path = Path(settings_path).resolve() if settings_path else None

    if not image_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {image_path}")
    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Input must be TIFF for now. Got: {image_path}")
    if settings_path is not None and not settings_path.exists():
        raise FileNotFoundError(f"MicrobeJ settings file not found: {settings_path}")

    project_root = Path(project_root).resolve() if project_root else find_project_root(Path(__file__).resolve())
    fiji_dir = ensure_fiji_dir(project_root)
    fiji_exe = find_fiji_executable(fiji_dir)

    install_microbej_into_fiji(
        project_root=project_root,
        fiji_dir=fiji_dir,
        microbej_jar=microbej_jar,
        overwrite=overwrite_plugin,
    )

    experiment_name = experiment_name or image_path.stem
    one_out = output_dir / image_path.stem
    one_out.mkdir(parents=True, exist_ok=True)

    macro_text = make_microbej_macro(
        image_path=image_path,
        settings_path=settings_path,
        output_dir=one_out,
        experiment_name=experiment_name,
        debug=debug,
    )

    with tempfile.TemporaryDirectory(prefix="pft_microbej_") as td:
        macro_path = Path(td) / "run_microbej.ijm"
        macro_path.write_text(macro_text, encoding="utf-8")

        cmd = [str(fiji_exe), "--headless", "-macro", str(macro_path)]

        print("\n[MICROBEJ COMMAND]")
        print("  " + " ".join(cmd))
        print("[MICROBEJ MACRO]")
        print(macro_text.strip())

        res = subprocess.run(
            cmd,
            cwd=str(fiji_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    elapsed = time.time() - t0

    print("\n--- MicrobeJ STDOUT ---")
    print(res.stdout)
    if res.stderr.strip():
        print("\n--- MicrobeJ STDERR ---")
        print(res.stderr)

    if res.returncode != 0:
        raise RuntimeError(
            f"MicrobeJ failed for image:\n  {image_path}\n"
            f"Return code: {res.returncode}\n"
            f"STDERR:\n{res.stderr}"
        )

    return MicrobeJRunResult(
        image_path=image_path,
        output_dir=one_out,
        experiment_name=experiment_name,
        returncode=int(res.returncode),
        stdout=res.stdout,
        stderr=res.stderr,
        elapsed_s=elapsed,
    )


def collect_tiff_images(path: Path, recursive: bool = False) -> list[Path]:
    """Collect TIFF images from a single file or folder."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Selected file is not TIFF: {path}")
        return [path.resolve()]

    if not path.exists():
        raise FileNotFoundError(path)

    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = list(path.rglob("*.tif")) + list(path.rglob("*.tiff"))
    else:
        pattern_iter = list(path.glob("*.tif")) + list(path.glob("*.tiff"))

    images = sorted({p.resolve() for p in pattern_iter})
    if not images:
        raise FileNotFoundError(f"No TIFF images found in: {path}")
    return images


def run_microbej_batch(
    *,
    images: list[Path],
    output_root: Path,
    settings_path: Path | None,
    project_root: Path | None = None,
    microbej_jar: Path | None = None,
    debug: bool = False,
    timeout_s: int | None = None,
    continue_on_error: bool = True,
) -> tuple[list[MicrobeJRunResult], list[tuple[Path, str]]]:
    """Run MicrobeJ over many TIFF images."""
    results: list[MicrobeJRunResult] = []
    errors: list[tuple[Path, str]] = []

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images, start=1):
        print("\n" + "=" * 80)
        print(f"[MICROBEJ BATCH] {i}/{len(images)}")
        print(f"  image: {img}")
        try:
            r = run_microbej_on_tiff(
                image_path=img,
                output_dir=output_root,
                settings_path=settings_path,
                project_root=project_root,
                microbej_jar=microbej_jar,
                experiment_name=img.stem,
                debug=debug,
                timeout_s=timeout_s,
                overwrite_plugin=False,
            )
            results.append(r)
            print(f"[OK] output: {r.output_dir}")
        except Exception as e:
            msg = str(e)
            errors.append((img, msg))
            print(f"[ERROR] {img}\n{msg}")
            if not continue_on_error:
                raise

    return results, errors
