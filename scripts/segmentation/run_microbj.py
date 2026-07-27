from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    The lookup is based on this script's physical location and therefore does
    not depend on the current working directory. An explicit error is raised
    when the expected repository layout cannot be found.
    """
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT_GUESS = _pft_project_root(THIS_FILE)
SRC = PROJECT_ROOT_GUESS / "src"

from PFT.core_prog_parts.segmentation.microbj import (  # noqa: E402
    collect_tiff_images,
    find_project_root,
    run_microbej_batch,
)


"""
Interactive runner for MicrobeJ on TIFF files.

"""


def _ask_path(prompt: str, default: Path | None = None, allow_empty: bool = False) -> Path | None:
    if default is not None:
        s = input(f"{prompt}\n[{default}]: ").strip()
        if s == "":
            return default
    else:
        s = input(f"{prompt}: ").strip()

    if s == "" and allow_empty:
        return None
    if s == "":
        raise SystemExit("No path entered.")
    return Path(s.strip('"')).expanduser()


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    s = input(f"{prompt} [{d}]: ").strip().lower()
    if s == "":
        return default
    return s in {"y", "yes", "1", "true", "t"}


def _print_images(images: list[Path]) -> None:
    print("\nSelected TIFF images:")
    for i, p in enumerate(images, start=1):
        print(f"  {i:03d}. {p}")


def main() -> int:
    project_root = find_project_root(Path(__file__).resolve())

    print("\nMicrobeJ TIFF runner")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print("This runner uses TIFF images only. OME-Zarr is intentionally not used here.\n")

    print("Choose input mode:")
    print("  1 = one TIFF file")
    print("  2 = all TIFF files in one folder")
    print("  3 = all TIFF files in one folder recursively")
    mode = input("Mode [2]: ").strip() or "2"

    default_tiff_dir = project_root / "results" / "img"
    selected = _ask_path("Choose TIFF file or folder", default=default_tiff_dir)
    if selected is None:
        raise SystemExit("No input selected.")

    recursive = mode == "3"
    if mode == "1":
        images = collect_tiff_images(selected, recursive=False)
    elif mode in {"2", "3"}:
        images = collect_tiff_images(selected, recursive=recursive)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    _print_images(images)
    if not _ask_yes_no("Run MicrobeJ on these images?", default=True):
        print("Stopped by user.")
        return 0

    settings = _ask_path(
        "Choose MicrobeJ settings file (.xml or .ini). Leave empty to use MicrobeJ default settings",
        default=None,
        allow_empty=True,
    )

    if settings is not None and settings.suffix.lower() not in {".xml", ".ini"}:
        raise SystemExit(f"Settings file should be .xml or .ini, got: {settings}")

    default_out = project_root / "results" / "microbej"
    out_root = _ask_path("Choose output root folder", default=default_out)
    if out_root is None:
        raise SystemExit("No output folder selected.")

    debug = _ask_yes_no("Enable MicrobeJ debug windows? Usually no", default=False)
    continue_on_error = _ask_yes_no("Continue if one image fails?", default=True)

    results, errors = run_microbej_batch(
        images=images,
        output_root=out_root,
        settings_path=settings,
        project_root=project_root,
        microbej_jar=None,
        debug=debug,
        timeout_s=None,
        continue_on_error=continue_on_error,
    )

    print("\n" + "=" * 80)
    print("MicrobeJ batch finished")
    print(f"  successful: {len(results)}")
    print(f"  failed    : {len(errors)}")
    print(f"  output    : {out_root}")

    if errors:
        print("\nFailed images:")
        for img, msg in errors:
            print(f"  - {img}")
            print(f"    {msg.splitlines()[0] if msg else 'unknown error'}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
