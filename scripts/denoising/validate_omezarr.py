"""
Revalidate existing OME-Zarr images against their original CZI files.
With no arguments, the script validates every ``*.ome.zarr`` directory below
``<project>/results/img``. ``--root`` may select another directory tree and
``--zarr`` may select one image. When a stored source path no longer exists,
``--raw-root`` relocates the CZI by filename or the script requests a folder in
interactive mode.

Every image receives a detailed ``ome_zarr_validation.txt`` beside the
OME-Zarr directory. Detailed checks are written only to that text file. Dataset summaries and
a global validation summary are refreshed after the scan.
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both 'scripts' and "
        f"'src/PFT/core_prog_parts'. Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"
if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))


import argparse
from pathlib import Path

from PFT.core_prog_parts.data_discovery import select_directory_dialog, stdin_is_interactive
from PFT.core_prog_parts.io import load_czi
from PFT.core_prog_parts.omezarr_validation import (
    ValidationSummaryRow,
    make_validation_summary_row,
    validate_ome_zarr,
    write_dataset_and_global_validation_summaries,
)


REPO_ROOT = _pft_project_root(Path(__file__).resolve())
DEFAULT_VALIDATION_ROOT = REPO_ROOT / "results" / "img"


def find_omezarr_directories(root: str | Path) -> list[Path]:
    """Return one or more OME-Zarr directories below a validation target."""
    target = Path(root).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Validation root does not exist: {target}")
    if target.is_dir() and target.name.lower().endswith(".ome.zarr"):
        return [target.resolve()]
    return sorted(
        (path.resolve() for path in target.rglob("*.ome.zarr") if path.is_dir()),
        key=lambda path: str(path).lower(),
    )


def stored_source_path(zarr_path: Path) -> Path | None:
    """Read the original CZI path stored in OME-Zarr root attributes."""
    try:
        import zarr
    except Exception as exc:
        raise ImportError("Validation requires zarr from the project dependencies.") from exc

    root = zarr.open_group(str(zarr_path), mode="r")
    value = root.attrs.get("source_path", "")
    if not value:
        pft_meta = root.attrs.get("pft_meta", {})
        if isinstance(pft_meta, dict):
            value = pft_meta.get("source_path", "")
    return Path(str(value)).expanduser() if value else None


def find_source_by_name(raw_root: Path, filename: str) -> list[Path]:
    """Search a raw-data root recursively for a CZI filename case-insensitively."""
    filename_lower = filename.lower()
    return sorted(
        (
            path.resolve()
            for path in raw_root.rglob("*")
            if path.is_file() and path.name.lower() == filename_lower
        ),
        key=lambda path: str(path).lower(),
    )


def choose_source(candidates: list[Path], *, interactive: bool) -> Path:
    """Select one source CZI from matching candidates."""
    if not candidates:
        raise FileNotFoundError("No matching CZI source file was found.")
    if len(candidates) == 1 or not interactive:
        return candidates[0]
    print("Multiple matching source files were found:")
    for index, path in enumerate(candidates):
        print(f"  [{index}] {path}")
    value = input("Select number (default 0): ").strip()
    index = int(value) if value else 0
    if not 0 <= index < len(candidates):
        raise IndexError(f"Selection must be within 0..{len(candidates) - 1}")
    return candidates[index]


def request_raw_root() -> Path:
    """Request the parent raw-data directory through a dialog or terminal input."""
    selected = select_directory_dialog("Select the parent folder containing original CZI data")
    if selected is None:
        value = input("Enter the parent raw-data folder: ").strip().strip('"')
        if not value:
            raise FileNotFoundError("No raw-data folder was supplied.")
        selected = Path(value).expanduser()
    if not selected.is_dir():
        raise FileNotFoundError(f"Raw-data folder not found: {selected}")
    return selected.resolve()


def resolve_source(
    zarr_path: Path,
    *,
    raw_root: Path | None,
    interactive: bool,
) -> tuple[Path, Path | None]:
    """Resolve the original CZI path and return the active relocation root."""
    stored = stored_source_path(zarr_path)
    if stored is not None and stored.is_file():
        return stored.resolve(), raw_root

    filename = stored.name if stored is not None and stored.name else f"{zarr_path.parent.name}.czi"
    active_root = raw_root
    if active_root is None and interactive:
        print(f"Original source unavailable for: {zarr_path}")
        print(f"Stored source path: {stored if stored is not None else '<missing>'}")
        active_root = request_raw_root()

    if active_root is None:
        raise FileNotFoundError(
            f"Source CZI is unavailable for {zarr_path}. Stored path: {stored}. "
            "Supply --raw-root to relocate source files by filename."
        )

    candidates = find_source_by_name(active_root, filename)
    return choose_source(candidates, interactive=interactive), active_root


def summary_root_for_target(scan_target: Path) -> Path:
    """Choose the directory under which dataset and global summaries are stored."""
    resolved = scan_target.resolve()
    default_root = DEFAULT_VALIDATION_ROOT.resolve()
    if resolved == default_root or resolved.is_relative_to(default_root):
        return default_root
    if resolved.name.lower().endswith(".ome.zarr"):
        return resolved.parent
    return resolved


def dataset_and_sample(zarr_path: Path, summary_root: Path) -> tuple[str, str]:
    """Derive compact dataset and sample identifiers from an OME-Zarr path."""
    sample_dir = zarr_path.parent.resolve()
    try:
        relative = sample_dir.relative_to(summary_root.resolve())
        dataset = relative.parts[0] if relative.parts else "unknown"
        sample = "/".join(relative.parts[1:]) if len(relative.parts) > 1 else sample_dir.name
    except ValueError:
        dataset = sample_dir.parent.name or "unknown"
        sample = sample_dir.name
    return dataset, sample


def build_parser() -> argparse.ArgumentParser:
    """Construct arguments for default-all, custom-root, or single-image validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate OME-Zarr images against original CZI files. With no source "
            "argument, all existing images below results/img are validated."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--root", type=str, help="Recursively scan for *.ome.zarr directories.")
    source.add_argument("--zarr", type=str, help="Validate one OME-Zarr directory.")
    parser.add_argument(
        "--raw-root",
        "--raw_root",
        dest="raw_root",
        type=str,
        default=None,
        help="Parent raw-data directory used when stored source paths no longer exist.",
    )
    parser.add_argument("--non-interactive", "--non_interactive", dest="non_interactive", action="store_true")
    parser.add_argument("--stop-on-error", "--stop_on_error", dest="stop_on_error", action="store_true")
    return parser


def main() -> None:
    """Validate all selected OME-Zarr images and refresh compact summaries."""
    args = build_parser().parse_args()
    scan_target = Path(args.zarr or args.root or DEFAULT_VALIDATION_ROOT).expanduser().resolve()
    zarr_paths = find_omezarr_directories(scan_target)
    if not zarr_paths:
        raise FileNotFoundError(f"No *.ome.zarr directories found below: {scan_target}")

    interactive = not args.non_interactive and stdin_is_interactive()
    raw_root = Path(args.raw_root).expanduser().resolve() if args.raw_root else None
    if raw_root is not None and not raw_root.is_dir():
        raise FileNotFoundError(f"--raw-root does not exist: {raw_root}")

    summary_root = summary_root_for_target(scan_target)
    extra_rows: list[ValidationSummaryRow] = []
    passed = 0
    failed = 0
    errors = 0

    print(f"Validating {len(zarr_paths)} existing OME-Zarr images below: {scan_target}")
    for index, zarr_path in enumerate(zarr_paths, start=1):
        print(f"  [{index}/{len(zarr_paths)}] {zarr_path.parent.name}")
        dataset, sample = dataset_and_sample(zarr_path, summary_root)
        try:
            source_path, raw_root = resolve_source(
                zarr_path,
                raw_root=raw_root,
                interactive=interactive,
            )
            arr, meta = load_czi(source_path)
            result = validate_ome_zarr(zarr_path, arr, meta, print_terminal=False)
            if result.passed:
                passed += 1
            else:
                failed += 1
                if args.stop_on_error:
                    break
        except Exception as exc:
            errors += 1
            extra_rows.append(make_validation_summary_row(
                dataset=dataset,
                sample=sample,
                status="ERROR",
                zarr_path=zarr_path,
                detail=f"{type(exc).__name__}: {exc}",
            ))
            if args.stop_on_error:
                break

    dataset_paths, global_path = write_dataset_and_global_validation_summaries(
        summary_root,
        extra_rows=extra_rows,
    )

    print("\nPFT OME-Zarr validation summary")
    print("=" * 72)
    print(f"OME-Zarr images selected: {len(zarr_paths)}")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"ERROR: {errors}")
    for dataset, path in sorted(dataset_paths.items()):
        print(f"Dataset summary [{dataset}]: {path}")
    print(f"Global summary: {global_path}")
    print("Detailed checks are stored only in each sample's ome_zarr_validation.txt.")


if __name__ == "__main__":
    main()
