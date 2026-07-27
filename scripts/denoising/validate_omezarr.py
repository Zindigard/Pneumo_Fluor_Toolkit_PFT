"""Command-line tool for revalidating existing CZI-derived OME-Zarr outputs.

The script scans one OME-Zarr directory or a directory tree, resolves the
corresponding original CZI file, performs exact pixel and metadata comparison,
and writes both per-image validation reports and a batch summary.
"""

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


import argparse
from datetime import datetime, timezone
from pathlib import Path

from PFT.core_prog_parts.data_discovery import select_directory_dialog, stdin_is_interactive
from PFT.core_prog_parts.io import load_czi
from PFT.core_prog_parts.omezarr_validation import validate_ome_zarr



def find_omezarr_directories(root: str | Path) -> list[Path]:
    """Return one or more ``*.ome.zarr`` directories below a validation target.
    
    A target that is itself an OME-Zarr directory is returned as a single item.
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Validation root does not exist: {root}")
    if root.is_dir() and root.name.lower().endswith(".ome.zarr"):
        return [root.resolve()]
    return sorted(
        (path.resolve() for path in root.rglob("*.ome.zarr") if path.is_dir()),
        key=lambda path: str(path).lower(),
    )


def stored_source_path(zarr_path: Path) -> Path | None:
    """Read the original CZI path stored in OME-Zarr root attributes.
    
    The function checks ``source_path`` first and then the embedded ``pft_meta``
    metadata as a compatibility fallback.
    """
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
    """Search a raw-data root recursively for files matching a CZI filename case-insensitively."""
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
    """Select one source CZI from matching candidates, optionally using a terminal prompt."""
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
    """Request the parent raw-data directory through a file dialog or terminal input."""
    selected = select_directory_dialog("Select the parent folder containing the original CZI data")
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
    """Resolve the original CZI file for an OME-Zarr image.
    
    The stored absolute path is used when valid. Otherwise, the function relocates
    the source by filename below ``raw_root`` and returns the active raw-data root.
    """
    stored = stored_source_path(zarr_path)
    if stored is not None and stored.is_file():
        return stored.resolve(), raw_root

    filename = stored.name if stored is not None and stored.name else f"{zarr_path.parent.name}.czi"
    active_root = raw_root
    if active_root is None and interactive:
        print(f"Original source is unavailable for: {zarr_path}")
        print(f"Stored source path: {stored if stored is not None else '<missing>'}")
        active_root = request_raw_root()

    if active_root is None:
        raise FileNotFoundError(
            f"Source CZI is unavailable for {zarr_path}. Stored path: {stored}. "
            "Supply --raw-root to relocate source files by filename."
        )

    candidates = find_source_by_name(active_root, filename)
    return choose_source(candidates, interactive=interactive), active_root


def write_summary(root: Path, rows: list[tuple[Path, str, str]]) -> Path:
    """Write a batch validation summary containing status, OME-Zarr path, and source or error text."""
    summary_path = root / "ome_zarr_validation_summary.txt" if root.is_dir() else root.parent / "ome_zarr_validation_summary.txt"
    lines = [
        "PFT OME-Zarr batch validation summary",
        "=" * 100,
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"OME-Zarr directories checked: {len(rows)}",
        "",
        f"{'STATUS':<8} | OME-ZARR | SOURCE/ERROR",
        "-" * 100,
    ]
    for zarr_path, status, source_or_error in rows:
        lines.append(f"{status:<8} | {zarr_path} | {source_or_error}")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line arguments for single-image or recursive OME-Zarr validation."""
    parser = argparse.ArgumentParser(
        description="Validate existing OME-Zarr files against their original CZI data."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=str, help="Recursively scan for *.ome.zarr directories.")
    source.add_argument("--zarr", type=str, help="Validate one OME-Zarr directory.")
    parser.add_argument(
        "--raw-root", "--raw_root",
        dest="raw_root",
        type=str,
        default=None,
        help="Parent raw-data directory used when stored source paths no longer exist.",
    )
    parser.add_argument("--non-interactive", "--non_interactive", dest="non_interactive", action="store_true")
    parser.add_argument("--stop-on-error", "--stop_on_error", dest="stop_on_error", action="store_true")
    return parser


def main() -> None:
    """Execute batch revalidation and save the final summary report."""
    args = build_parser().parse_args()
    scan_target = Path(args.zarr or args.root).expanduser().resolve()
    zarr_paths = find_omezarr_directories(scan_target)
    if not zarr_paths:
        raise FileNotFoundError(f"No *.ome.zarr directories found below: {scan_target}")

    interactive = not args.non_interactive and stdin_is_interactive()
    raw_root = Path(args.raw_root).expanduser().resolve() if args.raw_root else None
    if raw_root is not None and not raw_root.is_dir():
        raise FileNotFoundError(f"--raw-root does not exist: {raw_root}")

    rows: list[tuple[Path, str, str]] = []
    for index, zarr_path in enumerate(zarr_paths, start=1):
        print(f"\n[{index}/{len(zarr_paths)}] Validating {zarr_path}")
        try:
            source_path, raw_root = resolve_source(
                zarr_path,
                raw_root=raw_root,
                interactive=interactive,
            )
            arr, meta = load_czi(source_path)
            result = validate_ome_zarr(zarr_path, arr, meta, print_terminal=True)
            status = "PASS" if result.passed else "FAIL"
            rows.append((zarr_path, status, str(source_path)))
            if not result.passed and args.stop_on_error:
                break
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {message}")
            rows.append((zarr_path, "ERROR", message))
            if args.stop_on_error:
                break

    summary_root = scan_target if scan_target.is_dir() and not scan_target.name.lower().endswith(".ome.zarr") else scan_target.parent
    summary_path = write_summary(summary_root, rows)
    passed = sum(status == "PASS" for _, status, _ in rows)
    failed = len(rows) - passed
    print(f"\nValidation complete. PASS: {passed} | FAIL/ERROR: {failed}")
    print(f"Batch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
