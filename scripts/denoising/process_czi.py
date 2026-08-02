"""
Import CZI datasets, create compact exports, and validate OME-Zarr data.
The script supports three canonical datasets: ``2d_time``, ``2d_wga_dapi``,
and ``3d``. A user can process one dataset at a time or select ``all`` to
process every dataset in one run. Raw-data directories are resolved from the
configured defaults, a supplied parent directory, a recursive search root, or
an interactive folder selection.

All image outputs are written below ``results/img``. For 2D data, each sample
contains metadata, ``image_norm16.tif``, ``image_norm16_rgb.tif``, two PNG
previews, ``image.ome.zarr``, and a detailed ``ome_zarr_validation.txt``.
At the end of a 

run, the script refreshes one compact validation summary per dataset and a
global summary in ``results/img``.


Examples
--------
Show all command-line parameters:

    python scripts/denoising/process_czi.py --help

Process every CZI file in the 2D time-course source directory:

    python scripts/denoising/process_czi.py \
        --dataset 2d_time \
        --folder E:/2D_data_time \
        --all \
        --non-interactive \
        --pyramid-max-layer 2 \
        --pyramid-downscale 2
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    Args:
        start (_PFTPath | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        _PFTPath: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _pft_project_root()
    """
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
from dataclasses import dataclass, field
import os
from pathlib import Path

from PFT.core_prog_parts.data_discovery import (
    group_czi_files_by_relative_parent,
    resolve_dataset_directory,
    stdin_is_interactive,
)
from PFT.core_prog_parts.io import list_czi_files, load_czi
from PFT.core_prog_parts.omezarr_validation import (
    REPORT_FILENAME,
    ValidationSummaryRow,
    make_validation_summary_row,
    scan_validation_reports,
    write_dataset_and_global_validation_summaries,
)
from PFT.core_prog_parts.save import export_2d, export_3d


REPO_ROOT = _pft_project_root(Path(__file__).resolve())
RESULTS_IMG_DIR = REPO_ROOT / "results" / "img"
DATASETS = ("2d_time", "2d_wga_dapi", "3d")


@dataclass(frozen=True)
class Paths:
    """Store optional default raw-data locations for the supported datasets."""

    data_2d_time: Path = Path(os.environ.get("PFT_DATA_2D_TIME", r"E:\2D_data_time"))
    data_2d_wga_dapi: Path = Path(
        os.environ.get("PFT_DATA_2D_WGA_DAPI", r"E:\2D_data_WGA_DAPI_DNA")
    )
    data_3d: Path = Path(os.environ.get("PFT_DATA_3D", r"E:\3d_data"))


@dataclass
class ProcessingStats:
    """Collect current-run processing counts and summary rows without detailed checks."""

    datasets_requested: list[str] = field(default_factory=list)
    files_discovered: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    extra_summary_rows: list[ValidationSummaryRow] = field(default_factory=list)

    def merge(self, other: "ProcessingStats") -> None:
        """Add another processing result to this aggregate.

        Args:
            other ("ProcessingStats"): Value specifying other for the operation.

        Example:
            >>> instance = ProcessingStats(...)
            >>> instance.merge(other=...)
        """
        for dataset in other.datasets_requested:
            if dataset not in self.datasets_requested:
                self.datasets_requested.append(dataset)
        self.files_discovered += other.files_discovered
        self.files_succeeded += other.files_succeeded
        self.files_failed += other.files_failed
        self.extra_summary_rows.extend(other.extra_summary_rows)


PATHS = Paths()


def default_folder(dataset: str) -> Path:
    """Return the configured default raw-data directory for one dataset.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = default_folder(dataset="2d_time")
    """
    if dataset == "2d_time":
        return PATHS.data_2d_time
    if dataset == "2d_wga_dapi":
        return PATHS.data_2d_wga_dapi
    if dataset == "3d":
        return PATHS.data_3d
    raise ValueError(f"Unknown dataset: {dataset}")


def prompt_float(prompt: str, default: float) -> float:
    """Read a floating-point value or return the supplied default.

    Args:
        prompt (str): Text value specifying prompt.
        default (float): Numerical value controlling default.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = prompt_float(prompt="prompt", default=0.5)
    """
    value = input(f"{prompt} (empty={default}): ").strip()
    return float(value) if value else default


def prompt_int(prompt: str, default: int) -> int:
    """Read an integer or return the supplied default.

    Args:
        prompt (str): Text value specifying prompt.
        default (int): Numerical value controlling default.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = prompt_int(prompt="prompt", default=1)
    """
    value = input(f"{prompt} (empty={default}): ").strip()
    return int(value) if value else default


def prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    """Display numbered options and return the selected zero-based index.

    Args:
        prompt (str): Text value specifying prompt.
        options (list[str]): Text value specifying options.
        default (int): Numerical value controlling default. Defaults to ``0``.

    Returns:
        int: Computed numerical result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = prompt_choice(prompt="prompt", options="options")
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
    """Display discovered CZI files and return one selected source file.

    Args:
        files (list[Path]): Filesystem path used for files.
        root (Path): Root directory used to resolve relative project paths.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = choose_file_interactive(files=Path("path/to/resource"), root=Path("path/to/resource"))
    """
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
    """Build an output path while preserving nested experiment directories.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_file (Path): Filesystem path associated with source.
        dataset_root (Path): Directory used for dataset.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = output_dataset_path(
        ...     dataset="2d_time",
        ...     source_file=Path("path/to/resource"),
        ...     dataset_root=Path("path/to/resource"),
        ... )
    """
    try:
        relative_parent = source_file.resolve().parent.relative_to(dataset_root.resolve())
    except ValueError:
        relative_parent = Path()
    return Path(dataset) / relative_parent


def expected_2d_output_dir(dataset: str, source_file: Path, dataset_root: Path) -> Path:
    """Return the expected sample output directory for one 2D source image.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_file (Path): Filesystem path associated with source.
        dataset_root (Path): Directory used for dataset.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = expected_2d_output_dir(
        ...     dataset="2d_time",
        ...     source_file=Path("path/to/resource"),
        ...     dataset_root=Path("path/to/resource"),
        ... )
    """
    return RESULTS_IMG_DIR / output_dataset_path(dataset, source_file, dataset_root) / source_file.stem


def skipped_or_error_row(
    *,
    dataset: str,
    sample: str,
    source_file: Path | str,
    output_dir: Path,
    status: str,
    detail: str,
) -> ValidationSummaryRow:
    """Create a compact row when validation was skipped or processing failed.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.
        source_file (Path | str): Filesystem path associated with source.
        output_dir (Path): Directory where generated resources are written.
        status (str): Text value specifying status.
        detail (str): Text value specifying detail.

    Returns:
        ValidationSummaryRow: Result produced by the operation.

    Example:
        >>> result = skipped_or_error_row(
        ...     dataset="2d_time",
        ...     sample="sample",
        ...     source_file=Path("path/to/resource"),
        ...     output_dir=Path("path/to/resource"),
        ...     status="status",
        ...     detail="detail",
        ... )
    """
    return make_validation_summary_row(
        dataset=dataset,
        sample=sample,
        status=status,
        source_path=source_file,
        zarr_path=output_dir / "image.ome.zarr",
        detail=detail,
    )


def run_one_2d(
    dataset: str,
    source_file: Path,
    dataset_root: Path,
    *,
    visualize: bool,
    scalebar_um: float,
    validate: bool = True,
) -> tuple[Path, ProcessingStats]:
    """Load and export one 2D CZI image and return compact processing statistics.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_file (Path): Filesystem path associated with source.
        dataset_root (Path): Directory used for dataset.
        visualize (bool): Boolean flag controlling visualize.
        scalebar_um (float): Numerical value controlling scalebar um.
        validate (bool): Boolean flag controlling validate. Defaults to ``True``.

    Returns:
        tuple[Path, ProcessingStats]: Resolved or generated filesystem path.

    Example:
        >>> result = run_one_2d(
        ...     dataset="2d_time",
        ...     source_file=Path("path/to/resource"),
        ...     dataset_root=Path("path/to/resource"),
        ...     visualize=True,
        ...     scalebar_um=0.5,
        ... )
    """
    stats = ProcessingStats(datasets_requested=[dataset], files_discovered=1)
    output_dir = expected_2d_output_dir(dataset, source_file, dataset_root)
    try:
        arr, meta = load_czi(source_file)
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
        stats.files_succeeded = 1
        if not validate:
            stats.extra_summary_rows.append(skipped_or_error_row(
                dataset=dataset,
                sample=str(source_file.stem),
                source_file=source_file,
                output_dir=out_dir,
                status="SKIPPED",
                detail="automatic validation disabled by --skip-validation",
            ))
        return out_dir, stats
    except Exception as exc:
        stats.files_failed = 1
        report_path = output_dir / REPORT_FILENAME
        if not report_path.exists():
            stats.extra_summary_rows.append(skipped_or_error_row(
                dataset=dataset,
                sample=str(source_file.stem),
                source_file=source_file,
                output_dir=output_dir,
                status="ERROR",
                detail=f"{type(exc).__name__}: {exc}",
            ))
        raise


def run_all_2d(
    dataset: str,
    folder: Path,
    scalebar_um: float,
    *,
    start: int = 0,
    limit: int | None = None,
    stop_on_error: bool = False,
    validate: bool = True,
) -> ProcessingStats:
    """Process all recursively discovered 2D CZI files in one dataset.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        folder (Path): Filesystem path used for folder.
        scalebar_um (float): Numerical value controlling scalebar um.
        start (int): Numerical value controlling start. Defaults to ``0``.
        limit (int | None): Numerical value controlling limit. ``None`` selects the function's default behavior.
        stop_on_error (bool): Boolean flag controlling stop on error. Defaults to ``False``.
        validate (bool): Boolean flag controlling validate. Defaults to ``True``.

    Returns:
        ProcessingStats: Result produced by the operation.

    Example:
        >>> result = run_all_2d(
        ...     dataset="2d_time",
        ...     folder=Path("path/to/resource"),
        ...     scalebar_um=0.5,
        ... )
    """
    files = list_czi_files(folder, recursive=True)[start:]
    if limit is not None:
        files = files[:limit]
    stats = ProcessingStats(datasets_requested=[dataset], files_discovered=len(files))
    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"
    print(f"\nProcessing {dataset}: {len(files)} CZI files")

    for index, source_file in enumerate(files, start=1):
        print(f"  [{index}/{len(files)}] {source_file.name}")
        output_dir = expected_2d_output_dir(dataset, source_file, folder)
        try:
            arr, meta = load_czi(source_file)
            out_dir = export_2d(
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
            stats.files_succeeded += 1
            if not validate:
                stats.extra_summary_rows.append(skipped_or_error_row(
                    dataset=dataset,
                    sample=str(source_file.relative_to(folder).with_suffix("")),
                    source_file=source_file,
                    output_dir=out_dir,
                    status="SKIPPED",
                    detail="automatic validation disabled by --skip-validation",
                ))
        except Exception as exc:
            stats.files_failed += 1
            if not (output_dir / REPORT_FILENAME).exists():
                stats.extra_summary_rows.append(skipped_or_error_row(
                    dataset=dataset,
                    sample=str(source_file.relative_to(folder).with_suffix("")),
                    source_file=source_file,
                    output_dir=output_dir,
                    status="ERROR",
                    detail=f"{type(exc).__name__}: {exc}",
                ))
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            if stop_on_error:
                raise
    return stats


def run_all_3d(
    base_folder: Path,
    *,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
    stop_on_error: bool = False,
    validate: bool = True,
) -> ProcessingStats:
    """Process every recursively discovered 3D CZI file and preserve group paths.

    Args:
        base_folder (Path): Filesystem path used for base folder.
        pyramid_max_layer (int): Numerical value controlling pyramid max layer. Defaults to ``2``.
        pyramid_downscale (int): Numerical value controlling pyramid downscale. Defaults to ``2``.
        stop_on_error (bool): Boolean flag controlling stop on error. Defaults to ``False``.
        validate (bool): Boolean flag controlling validate. Defaults to ``True``.

    Returns:
        ProcessingStats: Result produced by the operation.

    Example:
        >>> result = run_all_3d(base_folder=Path("path/to/resource"))
    """
    groups = group_czi_files_by_relative_parent(base_folder)
    total = sum(len(files) for files in groups.values())
    stats = ProcessingStats(datasets_requested=["3d_data"], files_discovered=total)
    print(f"\nProcessing 3d: {total} CZI files in {len(groups)} groups")

    current = 0
    for group_name, files in groups.items():
        for source_file in files:
            current += 1
            print(f"  [{current}/{total}] {group_name}/{source_file.name}")
            output_dir = RESULTS_IMG_DIR / "3d_data" / group_name / source_file.stem
            try:
                arr, meta = load_czi(source_file)
                out_dir = export_3d(
                    arr=arr,
                    meta=meta,
                    dataset_folder=group_name,
                    out_base=RESULTS_IMG_DIR,
                    save_omezarr=True,
                    overwrite_omezarr=True,
                    validate_omezarr=validate,
                    pyramid_max_layer=pyramid_max_layer,
                    pyramid_downscale=pyramid_downscale,
                )
                stats.files_succeeded += 1
                if not validate:
                    stats.extra_summary_rows.append(skipped_or_error_row(
                        dataset="3d_data",
                        sample=f"{group_name}/{source_file.stem}",
                        source_file=source_file,
                        output_dir=out_dir,
                        status="SKIPPED",
                        detail="automatic validation disabled by --skip-validation",
                    ))
            except Exception as exc:
                stats.files_failed += 1
                if not (output_dir / REPORT_FILENAME).exists():
                    stats.extra_summary_rows.append(skipped_or_error_row(
                        dataset="3d_data",
                        sample=f"{group_name}/{source_file.stem}",
                        source_file=source_file,
                        output_dir=output_dir,
                        status="ERROR",
                        detail=f"{type(exc).__name__}: {exc}",
                    ))
                print(f"    ERROR: {type(exc).__name__}: {exc}")
                if stop_on_error:
                    raise
    return stats


def resolve_input_folder(
    dataset: str,
    *,
    supplied_folder: str | None,
    search_root: str | None,
    interactive: bool,
) -> Path:
    """Resolve one dataset folder from defaults, parent paths, or user selection.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        supplied_folder (str | None): Text value specifying supplied folder.
        search_root (str | None): Directory used for search.
        interactive (bool): Boolean flag controlling interactive.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = resolve_input_folder(
        ...     dataset="2d_time",
        ...     supplied_folder="supplied_folder",
        ...     search_root=Path("path/to/resource"),
        ...     interactive=True,
        ... )
    """
    preferred = Path(supplied_folder) if supplied_folder else default_folder(dataset)
    return resolve_dataset_directory(
        dataset,
        preferred=preferred,
        search_root=Path(search_root) if search_root else None,
        interactive=interactive,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line arguments for one-dataset or all-dataset processing.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """
    parser = argparse.ArgumentParser(
        description="Convert CZI datasets to OME-Zarr and verify exact data/metadata preservation."
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"])
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Dataset directory or a parent directory containing multiple datasets.",
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default=None,
        help="Search recursively below this path for recognized dataset folders.",
    )
    parser.add_argument("--all", action="store_true", help="Process all files in one selected 2D dataset.")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-on-error", "--stop_on_error", dest="stop_on_error", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--scalebar-um", "--scalebar_um", dest="scalebar_um", type=float, default=5.0)
    parser.add_argument("--pyramid-max-layer", "--pyramid_max_layer", dest="pyramid_max_layer", type=int, default=2)
    parser.add_argument("--pyramid-downscale", "--pyramid_downscale", dest="pyramid_downscale", type=int, default=2)
    parser.add_argument(
        "--non-interactive",
        "--non_interactive",
        dest="non_interactive",
        action="store_true",
        help="Do not request a replacement folder when raw data cannot be found.",
    )
    parser.add_argument(
        "--skip-validation",
        "--skip_validation",
        dest="skip_validation",
        action="store_true",
        help="Disable validation and mark outputs as SKIPPED in summaries.",
    )
    return parser


def _print_final_summary(stats: ProcessingStats, global_summary: Path, dataset_summaries: dict[str, Path]) -> None:
    """Print only aggregate processing and validation information at the end.

    Args:
        stats (ProcessingStats): Value specifying stats for the operation.
        global_summary (Path): Filesystem path used for global summary.
        dataset_summaries (dict[str, Path]): Filesystem path used for dataset summaries.

    Example:
        >>> _print_final_summary(
        ...     stats=...,
        ...     global_summary=Path("path/to/resource"),
        ...     dataset_summaries="dataset_summaries",
        ... )
    """
    all_rows = scan_validation_reports(RESULTS_IMG_DIR)
    pass_count = sum(row.status == "PASS" for row in all_rows)
    fail_count = sum(row.status == "FAIL" for row in all_rows)
    print("\nPFT processing summary")
    print("=" * 72)
    print(f"Datasets requested: {', '.join(stats.datasets_requested) or '-'}")
    print(f"CZI files discovered: {stats.files_discovered}")
    print(f"Files processed successfully: {stats.files_succeeded}")
    print(f"Files failed during this run: {stats.files_failed}")
    print(f"Existing detailed validation reports: {len(all_rows)}")
    print(f"Existing validation PASS: {pass_count}")
    print(f"Existing validation FAIL: {fail_count}")
    for dataset, path in sorted(dataset_summaries.items()):
        print(f"Dataset summary [{dataset}]: {path}")
    print(f"Global summary: {global_summary}")


def main() -> None:
    """Execute one selected dataset or all datasets and refresh all summaries.

    Raises:
        IndexError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    args = build_parser().parse_args()
    interactive = not args.non_interactive and stdin_is_interactive()
    validate = not args.skip_validation

    if args.dataset is None:
        options = ["2d_time", "2d_wga_dapi", "3d", "Process ALL datasets"]
        selected = prompt_choice("\nSelect processing mode:", options, 0)
        dataset = "all" if selected == 3 else DATASETS[selected]
        interactive = True
    else:
        dataset = args.dataset

    scalebar_um = args.scalebar_um
    if args.dataset is None and dataset in {"2d_time", "2d_wga_dapi", "all"}:
        scalebar_um = prompt_float("Scalebar length in µm", default=5.0)

    aggregate = ProcessingStats()
    selected_datasets = DATASETS if dataset == "all" else (dataset,)

    for current_dataset in selected_datasets:
        try:
            all_dataset_parent = args.folder if dataset == "all" else None
            folder = resolve_input_folder(
                current_dataset,
                supplied_folder=None if all_dataset_parent else args.folder,
                search_root=args.search_root or all_dataset_parent,
                interactive=interactive,
            )
        except Exception as exc:
            aggregate.datasets_requested.append("3d_data" if current_dataset == "3d" else current_dataset)
            aggregate.files_failed += 1
            aggregate.extra_summary_rows.append(make_validation_summary_row(
                dataset="3d_data" if current_dataset == "3d" else current_dataset,
                sample="<dataset discovery>",
                status="ERROR",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            print(f"\nCould not resolve {current_dataset}: {type(exc).__name__}: {exc}")
            if args.stop_on_error or dataset != "all":
                raise
            continue

        if current_dataset in {"2d_time", "2d_wga_dapi"}:
            process_all = args.all or dataset == "all"
            if args.dataset is None and dataset != "all":
                process_all = bool(prompt_choice("\nMode:", ["Process ONE file", "Process ALL files"], 0))

            files = list_czi_files(folder, recursive=True)
            if process_all:
                stats = run_all_2d(
                    current_dataset,
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
                out_dir, stats = run_one_2d(
                    current_dataset,
                    source_file,
                    folder,
                    visualize=args.visualize,
                    scalebar_um=scalebar_um,
                    validate=validate,
                )
                print(f"Saved sample: {out_dir}")
        else:
            stats = run_all_3d(
                folder,
                pyramid_max_layer=args.pyramid_max_layer,
                pyramid_downscale=args.pyramid_downscale,
                stop_on_error=args.stop_on_error,
                validate=validate,
            )
        aggregate.merge(stats)

    dataset_paths, global_path = write_dataset_and_global_validation_summaries(
        RESULTS_IMG_DIR,
        extra_rows=aggregate.extra_summary_rows,
    )
    _print_final_summary(aggregate, global_path, dataset_paths)


if __name__ == "__main__":
    main()
