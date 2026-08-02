r"""Provide command-line and programmatic utilities for create results statistics.

Examples
--------
Show all command-line parameters:

    python scripts/statistics/create_results_statistics.py --help

Representative execution:

    python scripts/statistics/create_results_statistics.py \
        --dataset 2d_time \
        --csv results/statistics/example.csv \
        --graph axial \
        --test welch
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
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _pft_project_root(THIS_FILE)
SRC_DIR = PROJECT_ROOT / "src"

from PFT.core_prog_parts.statistics.results_graphs import (  # noqa: E402
    find_project_root,
    list_csv_files,
)
from PFT.core_prog_parts.statistics.results_statistics import run_statistics  # noqa: E402


DATASETS = {
    "1": "2d_time_lapse_hada",
    "2": "2d_wga_dapi",
    "3": "3d_sim",
    "4": "segmentation",
}

GRAPH_TYPES = {
    "1": "cell_count",
    "2": "cell_length",
    "3": "cell_area",
    "4": "cell_volume",
    "5": "homogeneity",
    "6": "axial_profile",
    "7": "radial_profile",
    "8": "kymograph",
    "9": "colocalization",
    "10": "iou",
    "11": "dice",
}

TEST_TYPES = {
    "1": "t_test",
    "2": "two_way_anova",
    "3": "all",
}

DATASET_HINTS = {
    "2d_time_lapse_hada": ["time", "hada", "timelapse", "2d_time"],
    "2d_wga_dapi": ["wga", "dapi", "wagi"],
    "3d_sim": ["3d", "sim", "hada", "nada", "tada"],
    "segmentation": ["iou", "dice", "segmentation", "model"],
}


def choose_from_list(title: str, options: list[str]) -> str:
    """Choose from list according to the configured criteria.

    Args:
        title (str): Title displayed on the generated figure or report section.
        options (list[str]): Text value specifying options.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = choose_from_list(title="title", options="options")
    """
    print(f"\n{title}")
    for idx, item in enumerate(options, start=1):
        print(f"  {idx}. {item}")
    while True:
        value = input("Choose number: ").strip()
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1]
        print("Invalid choice. Try again.")


def choose_dataset() -> str:
    """Choose dataset according to the configured criteria.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = choose_dataset()
    """
    print("\nChoose dataset:")
    for key, value in DATASETS.items():
        print(f"  {key}. {value}")
    while True:
        value = input("Dataset number: ").strip()
        if value in DATASETS:
            return DATASETS[value]
        print("Invalid dataset choice.")


def choose_graph() -> str:
    """Choose graph according to the configured criteria.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = choose_graph()
    """
    print("\nChoose graph/measurement type:")
    for key, value in GRAPH_TYPES.items():
        print(f"  {key}. {value}")
    while True:
        value = input("Graph number: ").strip()
        if value in GRAPH_TYPES:
            return GRAPH_TYPES[value]
        print("Invalid graph choice.")


def choose_test() -> str:
    """Choose test according to the configured criteria.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = choose_test()
    """
    print("\nChoose statistical test:")
    for key, value in TEST_TYPES.items():
        print(f"  {key}. {value}")
    while True:
        value = input("Test number: ").strip()
        if value in TEST_TYPES:
            return TEST_TYPES[value]
        print("Invalid test choice.")


def score_csv_for_dataset(path: Path, dataset: str) -> int:
    """Return score CSV data for dataset for the supplied inputs.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = score_csv_for_dataset(path=Path("path/to/resource"), dataset="2d_time")
    """
    name = path.name.lower()
    return sum(1 for hint in DATASET_HINTS.get(dataset, []) if hint in name)


def choose_csv(project_root: Path, dataset: str) -> Path:
    """Choose CSV data according to the configured criteria.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = choose_csv(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    csvs = list_csv_files(project_root)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {project_root / 'results' / 'final_values'}.")

    csvs = sorted(csvs, key=lambda p: (-score_csv_for_dataset(p, dataset), str(p)))
    labels = [str(p.relative_to(project_root)) for p in csvs]
    choice = choose_from_list("Choose CSV file:", labels)
    return project_root / choice


def parse_value_cols(raw: str | None) -> list[str] | None:
    """Parse value cols into a validated representation.

    Args:
        raw (str | None): Text value specifying raw.

    Returns:
        list[str] | None: Collection containing the generated or selected values.

    Example:
        >>> result = parse_value_cols(raw="raw")
    """
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build parser from the supplied inputs.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """
    parser = argparse.ArgumentParser(
        description="Create thesis statistics tables from the same CSV files used for result graphs."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset", choices=list(DATASETS.values()), default=None)
    parser.add_argument("--graph", choices=list(GRAPH_TYPES.values()), default=None)
    parser.add_argument("--test", choices=list(TEST_TYPES.values()), default=None)
    parser.add_argument("--csv", type=Path, default=None)

    parser.add_argument("--value-col", default=None, help="Dependent variable column for non-profile tests.")
    parser.add_argument("--value-cols", default=None, help="Comma-separated profile/channel columns for AUC calculation.")
    parser.add_argument("--group-col", default=None, help="Condition/group column. Can also be medium/treatment.")
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--axis-col", default=None)
    parser.add_argument("--radius-col", default=None)
    parser.add_argument("--cell-col", default=None)
    parser.add_argument("--pair-col", default=None)
    parser.add_argument("--channel-col", default=None)

    parser.add_argument("--compare-col", default=None, help="Column used for pairwise Welch t-tests. If omitted, it is inferred.")
    parser.add_argument("--factor-a", default=None, help="First factor for two-way ANOVA. If omitted, it is inferred.")
    parser.add_argument("--factor-b", default=None, help="Second factor for two-way ANOVA. If omitted, it is inferred.")
    parser.add_argument("--use-auc", action="store_true", help="Force AUC calculation before running statistics.")
    parser.add_argument("--no-auc", action="store_true", help="Disable automatic AUC for axial/radial/kymograph graphs.")
    parser.add_argument("--no-3d-channel-switch", action="store_true", help="Do not apply the final blue/green switch for 3D SIM statistics.")
    parser.add_argument("--min-n", type=int, default=2, help="Minimum n per group for a t-test.")
    parser.add_argument("--stem", default=None, help="Output filename stem.")
    return parser


def main() -> None:
    """Execute the command-line workflow and return its process exit status.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    parser = build_parser()
    args = parser.parse_args()

    project_root = find_project_root(args.project_root)
    dataset = args.dataset or choose_dataset()
    graph = args.graph or choose_graph()
    test = args.test or choose_test()

    csv_path = args.csv
    if csv_path is None:
        csv_path = choose_csv(project_root, dataset)
    else:
        csv_path = csv_path if csv_path.is_absolute() else project_root / csv_path

    if args.use_auc and args.no_auc:
        raise ValueError("Use either --use-auc or --no-auc, not both.")
    use_auc = True if args.use_auc else False if args.no_auc else None

    print("\nCreating statistics")
    print(f"  Project root: {project_root}")
    print(f"  Dataset:      {dataset}")
    print(f"  Graph:        {graph}")
    print(f"  Test:         {test}")
    print(f"  CSV:          {csv_path}")

    result = run_statistics(
        dataset=dataset,
        graph=graph,
        csv_path=csv_path,
        project_root=project_root,
        test=test,
        value_col=args.value_col,
        value_cols=parse_value_cols(args.value_cols),
        group_col=args.group_col,
        time_col=args.time_col,
        axis_col=args.axis_col,
        radius_col=args.radius_col,
        cell_col=args.cell_col,
        pair_col=args.pair_col,
        channel_col=args.channel_col,
        compare_col=args.compare_col,
        factor_a=args.factor_a,
        factor_b=args.factor_b,
        use_auc=use_auc,
        switch_blue_green=not args.no_3d_channel_switch,
        stem=args.stem,
        min_n=args.min_n,
    )

    print("\nSaved statistics files:")
    print(f"  Primary CSV:  {result.primary_csv_path}")
    if result.prepared_csv_path:
        print(f"  Prepared CSV: {result.prepared_csv_path}")
    if result.extra_paths:
        print("  Extra files:")
        for path in result.extra_paths:
            print(f"    {path}")
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
