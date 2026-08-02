"""
Provide command-line and programmatic utilities for run statistics pipeline.
Run PCA, normalization and graph generation with one command.
Examples
--------
Show all command-line parameters:

    python scripts/statistics/run_statistics_pipeline.py --help

Representative 2D execution with at most 100 cells per ROI:

    python scripts/statistics/run_statistics_pipeline.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --exclude-border \
        --max-cells-per-roi-2d-time 100 \
        --selection-seed 1337 \
        --overwrite

Representative 3D execution with at most 100 cells per ROI:

    python scripts/statistics/run_statistics_pipeline.py \
        --dataset 3d_mip \
        --source-mode deconv_masked \
        --exclude-border \
        --max-cells-per-roi-3d 100 \
        --selection-seed 1337 \
        --overwrite
"""

from __future__ import annotations

"""Run PCA, normalization and graph generation with one command."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_FILE = Path(__file__).resolve()


def find_project_root(explicit: Path | None = None) -> Path:
    """Find project root in the available data or project structure.

    Args:
        explicit (Path | None): Filesystem path used for explicit. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_project_root()
    """
    if explicit is not None:
        return explicit.expanduser().resolve()
    current = SCRIPT_FILE.resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root; provide --project-root.")


def run(command: list[str], cwd: Path) -> None:
    """Run the requested operation using the supplied configuration.

    Args:
        command (list[str]): Text value specifying command.
        cwd (Path): Filesystem path used for cwd.

    Example:
        >>> run(command="command", cwd=Path("path/to/resource"))
    """
    print("\nRunning:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def build_parser() -> argparse.ArgumentParser:
    """Build parser from the supplied inputs.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA alignment, normalization and graph generation. Normal mode "
            "processes all eligible images; --example selects one low-cell image "
            "per condition."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--dataset",
        choices=("all", "2d_time", "2d_wga_dapi", "3d_mip"),
        default="all",
    )
    parser.add_argument("--source-mode")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--preferred-min-cells", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=80)
    parser.add_argument(
        "--max-cells-per-roi-2d-time",
        type=int,
        default=100,
        help=(
            "Maximum valid cells retained per 2d_time ROI during full-mode PCA. "
            "Use 0 to disable the cap. Default: 100."
        ),
    )
    parser.add_argument(
        "--max-cells-per-roi-3d",
        type=int,
        default=100,
        help=(
            "Maximum valid cells retained per 3d_mip ROI during full-mode PCA. "
            "Use 0 to disable the cap. Default: 100."
        ),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=1337,
        help="Base seed for reproducible per-ROI cell selection. Default: 1337.",
    )
    parser.add_argument("--exclude-time", type=int, action="append", default=[])
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--exclude-border", action="store_true")
    parser.add_argument("--target-length", type=int, default=90)
    parser.add_argument("--target-width", type=int, default=30)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--radial-bins", type=int, default=20)
    parser.add_argument(
        "--diagnostic-cells-per-roi",
        type=int,
        default=1,
        help=(
            "Maximum cells per ROI for which normalization before/after PNG "
            "figures are saved. Default: 1."
        ),
    )
    parser.add_argument("--pixel-size-um", type=float)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the command-line workflow and return its process exit status.

    Args:
        argv (Sequence[str] | None): Optional command-line argument sequence. When omitted, arguments are read from ``sys.argv``. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    args = build_parser().parse_args(argv)
    if args.diagnostic_cells_per_roi < 0:
        raise ValueError("--diagnostic-cells-per-roi must be 0 or greater")
    if args.max_cells_per_roi_2d_time < 0:
        raise ValueError("--max-cells-per-roi-2d-time must be 0 or greater")
    if args.max_cells_per_roi_3d < 0:
        raise ValueError("--max-cells-per-roi-3d must be 0 or greater")
    os.environ.setdefault("MPLBACKEND", "Agg")
    project_root = find_project_root(args.project_root)
    statistics_dir = project_root / "scripts" / "statistics"
    pca_script = statistics_dir / "align_cells_pca.py"
    normalization_script = statistics_dir / "normalize_aligned_cells.py"
    graph_script = statistics_dir / "create_statistics_graphs.py"
    common_module = statistics_dir / "statistics_pipeline_common.py"
    for path in (pca_script, normalization_script, graph_script, common_module):
        if not path.exists():
            raise FileNotFoundError(f"Required statistics pipeline file is missing: {path}")

    common_args = ["--dataset", args.dataset]
    if args.source_mode:
        common_args.extend(["--source-mode", args.source_mode])
    if args.example:
        common_args.append("--example")
    if args.skip_missing:
        common_args.append("--skip-missing")
    if args.overwrite:
        common_args.append("--overwrite")

    pca_command = [
        sys.executable,
        str(pca_script),
        *common_args,
        "--preferred-min-cells",
        str(args.preferred_min_cells),
        "--max-cells",
        str(args.max_cells),
        "--max-cells-per-roi-2d-time",
        str(args.max_cells_per_roi_2d_time),
        "--max-cells-per-roi-3d",
        str(args.max_cells_per_roi_3d),
        "--selection-seed",
        str(args.selection_seed),
    ]
    for time_value in args.exclude_time:
        pca_command.extend(["--exclude-time", str(time_value)])
    for sample in args.sample:
        pca_command.extend(["--sample", sample])
    if args.exclude_border:
        pca_command.append("--exclude-border")
    run(pca_command, project_root)

    normalization_command = [
        sys.executable,
        str(normalization_script),
        *common_args,
        "--target-length",
        str(args.target_length),
        "--target-width",
        str(args.target_width),
        "--margin",
        str(args.margin),
        "--diagnostic-cells-per-roi",
        str(args.diagnostic_cells_per_roi),
    ]
    run(normalization_command, project_root)

    graph_command = [
        sys.executable,
        str(graph_script),
        *common_args,
        "--radial-bins",
        str(args.radial_bins),
    ]
    if args.pixel_size_um is not None:
        graph_command.extend(["--pixel-size-um", str(args.pixel_size_um)])
    run(graph_command, project_root)

    print("\nStatistics pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
