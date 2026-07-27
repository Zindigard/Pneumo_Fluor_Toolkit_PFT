"""Command-line entry point for importing Zeiss CZI microscopy datasets.

The script locates 2D time-series, 2D WGA-DAPI, or 3D raw-data folders,
loads each CZI file, exports TIFF previews and OME-Zarr images, validates that
pixel values and metadata were preserved, and writes the outputs below the
project ``results`` directory.
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
from dataclasses import dataclass
import os
from pathlib import Path

from PFT.core_prog_parts.data_discovery import (
    group_czi_files_by_relative_parent,
    resolve_dataset_directory,
    stdin_is_interactive,
)
from PFT.core_prog_parts.io import list_czi_files, load_czi
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.omezarr_validation import validate_or_raise
from PFT.core_prog_parts.save import export_2d, export_3d


REPO_ROOT = _pft_project_root(Path(__file__).resolve())
TRAIN_COLLECTION_DIR = REPO_ROOT / "results" / "training_files"


@dataclass(frozen=True)
class Paths:
    """Store optional default raw-data locations for the supported datasets.
    
    Environment variables override the Windows defaults so the same script can be
    used on another workstation without modifying source code.
    """

    data_2d_time: Path = Path(os.environ.get("PFT_DATA_2D_TIME", r"E:\2D_data_time"))
    data_2d_wga_dapi: Path = Path(
        os.environ.get("PFT_DATA_2D_WGA_DAPI", r"E:\2D_data_WGA_DAPI_DNA")
    )
    data_3d: Path = Path(os.environ.get("PFT_DATA_3D", r"E:\3d_data"))


PATHS = Paths()


def default_folder(dataset: str) -> Path:
    """Return the configured default raw-data directory for a dataset identifier.
    
    Raises ``ValueError`` when the identifier is not one of ``2d_time``,
    ``2d_wga_dapi``, or ``3d``.
    """
    if dataset == "2d_time":
        return PATHS.data_2d_time
    if dataset == "2d_wga_dapi":
        return PATHS.data_2d_wga_dapi
    if dataset == "3d":
        return PATHS.data_3d
    raise ValueError(f"Unknown dataset: {dataset}")


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Read a yes-or-no response from the terminal and apply a default for empty input."""
    suffix = " (Y/n): " if default else " (y/N): "
    value = input(prompt + suffix).strip().lower()
    if not value:
        return default
    return value in ("y", "yes", "1", "true")


def prompt_float(prompt: str, default: float) -> float:
    """Read a floating-point value from the terminal or return the supplied default."""
    value = input(f"{prompt} (empty={default}): ").strip()
    return float(value) if value else default


def prompt_int(prompt: str, default: int) -> int:
    """Read an integer from the terminal or return the supplied default."""
    value = input(f"{prompt} (empty={default}): ").strip()
    return int(value) if value else default


def prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    """Display numbered options and return the selected zero-based index.
    
    Raises ``ValueError`` when the selected index is outside the available range.
    """
    print(prompt)
    for index, option in enumerate(options):
        marker = " (default)" if index == default else ""
        print(f"   [{index}] {option}{marker}")
    value = input("Enter number: ").strip()
    index = int(value) if value else default
    if not 0 <= index < len(options):
        raise ValueError("Invalid choice")
    return index


def choose_file_interactive(files: list[Path], root: Path) -> Path:
    """Display discovered CZI files relative to a dataset root and return the selected file."""
    print("\nAvailable CZI files:")
    for index, file_path in enumerate(files):
        try:
            label = file_path.relative_to(root)
        except ValueError:
            label = file_path
        print(f"[{index:03d}] {label}")
    index = prompt_int("Choose file index", default=0)
    if not 0 <= index < len(files):
        raise ValueError(f"Index {index} is outside 0..{len(files) - 1}")
    return files[index]




def output_dataset_path(dataset: str, source_file: Path, dataset_root: Path) -> Path:
    """Build the output dataset path while preserving nested experiment directories.
    
    Preserving the relative parent path prevents samples with identical filenames
    from overwriting each other.
    """
    try:
        relative_parent = source_file.resolve().parent.relative_to(dataset_root.resolve())
    except ValueError:
        relative_parent = Path()
    return Path(dataset) / relative_parent

def save_training_copy_and_validate(
    dataset: str,
    source_file: Path,
    dataset_root: Path,
    arr,
    meta,
    *,
    validate: bool,
) -> Path:
    """Write a single-scale training OME-Zarr copy and optionally validate it.
    
    Returns the path of the generated ``image.ome.zarr`` directory.
    """
    target_dir = TRAIN_COLLECTION_DIR / output_dataset_path(dataset, source_file, dataset_root) / source_file.stem
    zarr_path = save_ome_zarr_next_to_outputs(
        target_dir,
        arr,
        meta,
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        pyramid_downscale=2,
    )
    if validate:
        validate_or_raise(zarr_path, arr, meta, print_terminal=True)
    return zarr_path


def run_one_2d(
    dataset: str,
    source_file: Path,
    dataset_root: Path,
    *,
    visualize: bool,
    scalebar_um: float,
    validate: bool = True,
) -> Path:
    """Load and export one 2D CZI image, then validate all generated OME-Zarr copies.
    
    Returns the sample output directory under ``results/img``.
    """
    arr, meta = load_czi(source_file)
    print("\n=== CZI METADATA ===")
    print(meta)

    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"
    out_dir = export_2d(
        arr=arr,
        meta=meta,
        dataset_name=str(output_dataset_path(dataset, source_file, dataset_root)),
        preview_mode=preview_mode,
        visualize=visualize,
        save_preview_png=True,
        scalebar_um=scalebar_um,
        wga_ch=1,
        dapi_ch=0,
        validate_omezarr=validate,
    )
    save_training_copy_and_validate(
        dataset, source_file, dataset_root, arr, meta, validate=validate
    )
    return out_dir


def run_all_2d(
    dataset: str,
    folder: Path,
    scalebar_um: float,
    *,
    start: int = 0,
    limit: int | None = None,
    stop_on_error: bool = False,
    validate: bool = True,
) -> None:
    """Process all recursively discovered CZI files in a 2D dataset directory.
    
    The function supports start and limit slicing, counts successful and failed
    files, and can stop immediately after the first error.
    """
    files = list_czi_files(folder, recursive=True)[start:]
    if limit is not None:
        files = files[:limit]

    print(f"\nBatch processing {len(files)} files from {folder}. Dataset: {dataset}")
    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"

    successful = 0
    failed = 0
    for index, source_file in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {source_file}")
        try:
            arr, meta = load_czi(source_file)
            export_2d(
                arr=arr,
                meta=meta,
                dataset_name=str(output_dataset_path(dataset, source_file, folder)),
                preview_mode=preview_mode,
                visualize=False,
                save_preview_png=True,
                scalebar_um=scalebar_um,
                wga_ch=1,
                dapi_ch=0,
                validate_omezarr=validate,
            )
            save_training_copy_and_validate(
                dataset, source_file, folder, arr, meta, validate=validate
            )
            successful += 1
        except Exception as exc:
            failed += 1
            print(f"ERROR on {source_file.name}: {type(exc).__name__}: {exc}")
            if stop_on_error:
                raise

    print(f"\n2D processing complete. Success: {successful} | Failed: {failed}")


def run_all_3d(
    base_folder: Path,
    *,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
    stop_on_error: bool = False,
    validate: bool = True,
) -> None:
    """Group, export, and validate all recursively discovered 3D CZI files.
    
    The requested pyramid depth and lateral downsampling factor are forwarded to
    the OME-Zarr writer.
    """
    groups = group_czi_files_by_relative_parent(base_folder)
    out_base = REPO_ROOT / "results" / "img"
    successful = 0
    failed = 0

    print(f"\nDiscovered {sum(len(files) for files in groups.values())} CZI files in {len(groups)} groups.")
    for group_name, files in groups.items():
        print(f"\n=== 3D group: {group_name} | files: {len(files)} ===")
        for index, source_file in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {source_file}")
            try:
                arr, meta = load_czi(source_file)
                export_3d(
                    arr=arr,
                    meta=meta,
                    dataset_folder=group_name,
                    out_base=out_base,
                    save_omezarr=True,
                    overwrite_omezarr=True,
                    validate_omezarr=validate,
                    pyramid_max_layer=pyramid_max_layer,
                    pyramid_downscale=pyramid_downscale,
                )
                successful += 1
            except Exception as exc:
                failed += 1
                print(f"ERROR on {source_file.name}: {type(exc).__name__}: {exc}")
                if stop_on_error:
                    raise

    print(f"\n3D processing complete. Success: {successful} | Failed: {failed}")


def resolve_input_folder(
    dataset: str,
    *,
    supplied_folder: str | None,
    search_root: str | None,
    interactive: bool,
) -> Path:
    """Resolve the dataset folder from a supplied path, search root, or interactive selection."""
    preferred = Path(supplied_folder) if supplied_folder else default_folder(dataset)
    return resolve_dataset_directory(
        dataset,
        preferred=preferred,
        search_root=Path(search_root) if search_root else None,
        interactive=interactive,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for CZI import and validation options."""
    parser = argparse.ArgumentParser(
        description="Convert CZI datasets to OME-Zarr and verify exact data/metadata preservation."
    )
    parser.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi", "3d"])
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Dataset directory or a parent raw-data directory.",
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default=None,
        help="Search recursively below this path for a conventionally named dataset folder.",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-on-error", "--stop_on_error", dest="stop_on_error", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--scalebar-um", "--scalebar_um", dest="scalebar_um", type=float, default=5.0)
    parser.add_argument("--pyramid-max-layer", "--pyramid_max_layer", dest="pyramid_max_layer", type=int, default=2)
    parser.add_argument("--pyramid-downscale", "--pyramid_downscale", dest="pyramid_downscale", type=int, default=2)
    parser.add_argument(
        "--non-interactive", "--non_interactive",
        dest="non_interactive",
        action="store_true",
        help="Do not request a replacement folder when raw data cannot be found.",
    )
    parser.add_argument(
        "--skip-validation", "--skip_validation",
        dest="skip_validation",
        action="store_true",
        help="Disable automatic OME-Zarr validation. Not recommended for final data generation.",
    )
    return parser


def main() -> None:
    """Execute the complete command-line import workflow for the selected dataset."""
    args = build_parser().parse_args()
    interactive = not args.non_interactive and stdin_is_interactive()
    validate = not args.skip_validation

    if args.dataset is None:
        dataset = ["2d_time", "2d_wga_dapi", "3d"][
            prompt_choice("\nSelect dataset:", ["2d_time", "2d_wga_dapi", "3d"], 0)
        ]
        interactive = True
    else:
        dataset = args.dataset

    folder = resolve_input_folder(
        dataset,
        supplied_folder=args.folder,
        search_root=args.search_root,
        interactive=interactive,
    )

    if dataset in ("2d_time", "2d_wga_dapi"):
        scalebar_um = args.scalebar_um
        process_all = args.all
        if args.dataset is None:
            scalebar_um = prompt_float("Scalebar length in µm", default=5.0)
            process_all = bool(prompt_choice("\nMode:", ["Process ONE file", "Process ALL files"], 0))

        files = list_czi_files(folder, recursive=True)
        if process_all:
            run_all_2d(
                dataset,
                folder,
                scalebar_um,
                start=args.start,
                limit=args.limit,
                stop_on_error=args.stop_on_error,
                validate=validate,
            )
        else:
            if args.index is not None:
                if not 0 <= args.index < len(files):
                    raise IndexError(f"--index must be within 0..{len(files) - 1}")
                source_file = files[args.index]
            elif interactive:
                source_file = choose_file_interactive(files, folder)
            else:
                source_file = files[0]
            out_dir = run_one_2d(
                dataset,
                source_file,
                folder,
                visualize=args.visualize,
                scalebar_um=scalebar_um,
                validate=validate,
            )
            print(f"\nSaved to: {out_dir}")
    else:
        run_all_3d(
            folder,
            pyramid_max_layer=args.pyramid_max_layer,
            pyramid_downscale=args.pyramid_downscale,
            stop_on_error=args.stop_on_error,
            validate=validate,
        )


if __name__ == "__main__":
    main()
