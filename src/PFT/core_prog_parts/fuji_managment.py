from __future__ import annotations
import os
import zipfile
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

"""Helper functions to ensure Fiji and PSF Generator plugin are available for PSF generation in pipelines."""

FIJI_ZIP_URLS = [
    "https://downloads.imagej.net/fiji/latest/fiji-latest-win64-jdk.zip",
    "https://downloads.imagej.net/fiji/latest/fiji-latest-portable-nojava.zip",
]

PSFGEN_JAR_NAME = "PSF_Generator.jar"

PSFGEN_JAR_URLS = [
    "https://bigwww.epfl.ch/algorithms/psfgenerator/PSF_Generator.jar",
    "https://bigwww.epfl.ch/deconvolution/PSF_Generator.jar",
]


def find_project_root(start: Path) -> Path:
    """
    Find repository root by walking upward and looking for one of:
      - pyproject.toml
      - .git
      - setup.cfg
      - src
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "setup.cfg").exists():
            return p
        if (p / "src").exists():
            return p
    return start.parent


def get_cache_dir(project_root: Path) -> Path:
    return project_root / ".cache" / "fiji"


def find_fiji_dir(cache: Path) -> Path:
    """
    Fiji locator for Windows zips. Accepts any directory containing 'jars'
    and at least one known launcher file.
    """
    launcher_names = [
        "fiji-windows-x64",
        "fiji-windows-x64.exe",
        "ImageJ-win64.exe",
        "Fiji-win64.exe",
        "ImageJ.exe",
        "Fiji.exe",
        "fiji.bat",
    ]

    def is_fiji_dir(p: Path) -> bool:
        if not (p / "jars").is_dir():
            return False
        return any((p / name).exists() for name in launcher_names)

    candidates = [cache]
    if cache.exists():
        candidates += [p for p in cache.iterdir() if p.is_dir()]
        for p in list(candidates):
            candidates += [q for q in p.iterdir() if q.is_dir()]

    for p in candidates:
        if is_fiji_dir(p):
            return p

    if cache.exists():
        for jars_dir in cache.rglob("jars"):
            if jars_dir.is_dir():
                p = jars_dir.parent
                if is_fiji_dir(p):
                    return p

    raise RuntimeError("Fiji folder not found after extraction.")


def ensure_fiji_installed(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "fiji.zip"

    try:
        return find_fiji_dir(cache_dir)
    except Exception:
        pass

    last_err: Optional[Exception] = None
    for url in FIJI_ZIP_URLS:
        try:
            print("Downloading:", url, flush=True)
            urlretrieve(url, zip_path)
            print("Downloaded to:", zip_path, flush=True)
            last_err = None
            break
        except (HTTPError, URLError) as e:
            print("Download failed:", url, flush=True)
            print("Error:", e, flush=True)
            last_err = e

    if last_err is not None:
        raise RuntimeError(f"Could not download Fiji. Last error: {last_err}")

    print("Extracting...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    return find_fiji_dir(cache_dir)


def ensure_fiji_in_project(project_root: Path, quiet: bool = True) -> Path:
    """
    Convenience wrapper for pipelines.
    """
    configure_java_from_conda()
    cache_dir = get_cache_dir(project_root)
    fiji_dir = ensure_fiji_installed(cache_dir)
    if not quiet:
        print("Fiji ready at:", fiji_dir, flush=True)
    return fiji_dir


def psf_generator_exists(fiji_dir: Path) -> Path | None:
    """
    Returns the PSF_Generator.jar path if present, otherwise None.
    """
    jar_path = fiji_dir / "plugins" / PSFGEN_JAR_NAME
    if jar_path.exists() and jar_path.stat().st_size > 0:
        return jar_path
    return None


def ensure_psf_generator_exists(
    fiji_dir: Path,
    *,
    auto_download: bool = True,
    quiet: bool = True,
) -> Path:
    """
    Ensure PSF Generator exists in Fiji/plugins.

    If it exists: return jar path.
    If missing:
      - if auto_download=True: try to download it
      - otherwise: raise RuntimeError

    This function NEVER silently passes when missing.
    """
    plugins_dir = fiji_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    existing = psf_generator_exists(fiji_dir)
    if existing is not None:
        return existing

    jar_path = plugins_dir / PSFGEN_JAR_NAME

    if not auto_download:
        raise RuntimeError(
            "PSF Generator plugin is missing.\n"
            f"Place {PSFGEN_JAR_NAME} into:\n  {jar_path}\n"
            "Then rerun."
        )

    override = os.environ.get("PFT_PSFGEN_JAR_URL", "").strip()
    urls = [override] if override else list(PSFGEN_JAR_URLS)

    last_err: Optional[Exception] = None
    for url in urls:
        try:
            if not quiet:
                print("Downloading PSF Generator jar:", url, flush=True)
            urlretrieve(url, jar_path)
            if jar_path.exists() and jar_path.stat().st_size > 0:
                return jar_path
        except (HTTPError, URLError) as e:
            last_err = e
            continue

    try:
        if jar_path.exists() and jar_path.stat().st_size == 0:
            jar_path.unlink()
    except Exception:
        pass

    raise RuntimeError(
        "PSF Generator plugin is missing and automatic download failed.\n"
        f"Place {PSFGEN_JAR_NAME} into:\n  {jar_path}\n"
        "Or set environment variable PFT_PSFGEN_JAR_URL to a direct jar URL.\n"
        f"Last error: {last_err}"
    )


def configure_java_from_conda() -> None:
    """
    If running in conda on Windows, ensure JAVA_HOME points to env's Library.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        java_home = Path(conda_prefix) / "Library"
        os.environ.setdefault("JAVA_HOME", str(java_home))
        os.environ.setdefault("SJ_JAVA_HOME", str(java_home))


def main() -> int:
    configure_java_from_conda()

    here = Path(__file__).resolve()
    project_root = find_project_root(here)

    fiji_dir = ensure_fiji_in_project(project_root, quiet=False)

    ensure_psf_generator_exists(fiji_dir, auto_download=True, quiet=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())