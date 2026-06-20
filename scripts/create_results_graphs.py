
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "scripts" else Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.results_graphs import find_project_root, list_csv_files, run_graph  # noqa: E402


DATASETS = {
    "1": "2d_time_lapse_hada",
    "2": "2d_wga_dapi",
    "3": "3d_sim",
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
    "10": "correlation",
}

DATASET_HINTS = {
    "2d_time_lapse_hada": ["time", "hada", "timelapse", "2d_time"],
    "2d_wga_dapi": ["wga", "dapi", "wagi"],
    "3d_sim": ["3d", "sim", "hada", "nada", "tada"],
}


def choose_from_list(title: str, options: list[str]) -> str:
    print(f"\n{title}")
    for idx, item in enumerate(options, start=1):
        print(f"  {idx}. {item}")
    while True:
        value = input("Choose number: ").strip()
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1]
        print("Invalid choice. Try again.")


def choose_dataset() -> str:
    print("\nChoose dataset:")
    for key, value in DATASETS.items():
        print(f"  {key}. {value}")
    while True:
        value = input("Dataset number: ").strip()
        if value in DATASETS:
            return DATASETS[value]
        print("Invalid dataset choice.")


def choose_graph() -> str:
    print("\nChoose graph type:")
    for key, value in GRAPH_TYPES.items():
        print(f"  {key}. {value}")
    while True:
        value = input("Graph number: ").strip()
        if value in GRAPH_TYPES:
            return GRAPH_TYPES[value]
        print("Invalid graph choice.")


def score_csv_for_dataset(path: Path, dataset: str) -> int:
    name = path.name.lower()
    return sum(1 for hint in DATASET_HINTS.get(dataset, []) if hint in name)


def choose_csv(project_root: Path, dataset: str) -> Path:
    csvs = list_csv_files(project_root)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {project_root / 'results' / 'final_values'}.")

    csvs = sorted(csvs, key=lambda p: (-score_csv_for_dataset(p, dataset), str(p)))
    labels = [str(p.relative_to(project_root)) for p in csvs]
    choice = choose_from_list("Choose CSV file:", labels)
    return project_root / choice


def parse_value_cols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create thesis result graphs from existing CSV files.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset", choices=list(DATASETS.values()), default=None)
    parser.add_argument("--graph", choices=list(GRAPH_TYPES.values()), default=None)
    parser.add_argument("--csv", type=Path, default=None)

    parser.add_argument("--value-col", default=None)
    parser.add_argument("--value-cols", default=None, help="Comma-separated columns for line profiles.")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--axis-col", default=None)
    parser.add_argument("--radius-col", default=None)
    parser.add_argument("--cell-col", default=None)
    parser.add_argument("--pair-col", default=None)
    parser.add_argument("--x-col", default=None)
    parser.add_argument("--y-col", default=None)
    parser.add_argument("--stem", default=None, help="Output filename without extension.")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--intensity-label", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = find_project_root(args.project_root)
    dataset = args.dataset or choose_dataset()
    graph = args.graph or choose_graph()

    csv_path = args.csv
    if csv_path is None:
        csv_path = choose_csv(project_root, dataset)
    else:
        csv_path = csv_path if csv_path.is_absolute() else project_root / csv_path

    print("\nCreating graph")
    print(f"  Project root: {project_root}")
    print(f"  Dataset:      {dataset}")
    print(f"  Graph:        {graph}")
    print(f"  CSV:          {csv_path}")

    result = run_graph(
        dataset=dataset,
        graph=graph,
        csv_path=csv_path,
        project_root=project_root,
        value_col=args.value_col,
        value_cols=parse_value_cols(args.value_cols),
        group_col=args.group_col,
        time_col=args.time_col,
        axis_col=args.axis_col,
        radius_col=args.radius_col,
        cell_col=args.cell_col,
        pair_col=args.pair_col,
        x_col=args.x_col,
        y_col=args.y_col,
        stem=args.stem,
        ylabel=args.ylabel,
        intensity_label=args.intensity_label,
    )

    print("\nSaved graph files:")
    print(f"  PNG: {result.png_path}")
    print(f"  PDF: {result.pdf_path}")
    if result.csv_path:
        print(f"  CSV: {result.csv_path}")


if __name__ == "__main__":
    main()
