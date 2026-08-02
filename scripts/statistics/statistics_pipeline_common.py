r"""Provide command-line and programmatic utilities for statistics pipeline common.

Examples
--------
Programmatic use:

    from scripts.statistics.statistics_pipeline_common import find_project_root

    result = find_project_root(...)
"""

from __future__ import annotations

"""Shared dataset discovery and selection utilities for the statistics pipeline."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DATASETS: tuple[str, ...] = ("2d_time", "2d_wga_dapi", "3d_mip")


@dataclass(frozen=True)
class DatasetConfig:
    """Store validated configuration or result data for dataset config."""
    name: str
    default_source_mode: str
    channel_names: tuple[str, ...]
    condition_order: tuple[str, ...]


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "2d_time": DatasetConfig(
        name="2d_time",
        default_source_mode="filtered_unet",
        channel_names=("HADA",),
        condition_order=(
            "THY_5min",
            "NHS_5min",
            "THY_20min",
            "NHS_20min",
            "THY_40min",
            "NHS_40min",
            "THY_120min",
            "NHS_120min",
        ),
    ),
    "2d_wga_dapi": DatasetConfig(
        name="2d_wga_dapi",
        default_source_mode="filtered_unet",
        channel_names=("DAPI", "WGA"),
        condition_order=("THY_noCSP", "THY_CSP", "NHS_noCSP", "NHS_CSP"),
    ),
    "3d_mip": DatasetConfig(
        name="3d_mip",
        default_source_mode="deconv_masked",
        channel_names=("HADA", "NADA", "TADA"),
        condition_order=("THY", "NHS"),
    ),
}


def find_project_root(script_file: Path, explicit: Path | None = None) -> Path:
    """Find project root in the available data or project structure.

    Args:
        script_file (Path): Filesystem path associated with script.
        explicit (Path | None): Filesystem path used for explicit. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_project_root(script_file=Path("path/to/resource"))
    """
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / "scripts").is_dir():
            raise FileNotFoundError(f"Project root does not contain scripts/: {root}")
        return root

    current = script_file.resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root; provide --project-root.")


def selected_datasets(value: str) -> tuple[str, ...]:
    """Return selected datasets for the supplied inputs.

    Args:
        value (str): Value to validate, transform, store, or forward.

    Returns:
        tuple[str, ...]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = selected_datasets(value="value")
    """
    if value == "all":
        return DATASETS
    if value not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {value}")
    return (value,)


def source_mode_for(dataset: str, source_mode: str | None, dataset_argument: str) -> str:
    """Return source mode for for the supplied inputs.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str | None): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        dataset_argument (str): Text value specifying dataset argument.

    Returns:
        str: Generated or resolved text value.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = source_mode_for(
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     dataset_argument="dataset_argument",
        ... )
    """
    if source_mode is not None:
        if dataset_argument == "all":
            raise ValueError(
                "--source-mode can only be used when one dataset is selected. "
                "The default source mode is used for --dataset all."
            )
        return source_mode
    return DATASET_CONFIGS[dataset].default_source_mode


def infer_condition(dataset: str, sample_name: str) -> tuple[str, dict[str, object]]:
    """Infer condition from the supplied model inputs.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample_name (str): Text value specifying sample name.

    Returns:
        tuple[str, dict[str, object]]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = infer_condition(dataset="2d_time", sample_name="sample_name")
    """
    upper = sample_name.upper()
    medium = "THY" if "THY" in upper else "NHS" if "NHS" in upper else "UNKNOWN"
    metadata: dict[str, object] = {"medium": medium}

    if dataset == "2d_time":
        match = re.search(r"(?<!\d)(5|20|40|120)\s*MIN", upper)
        time_min = int(match.group(1)) if match else -1
        metadata["time_min"] = time_min
        return f"{medium}_{time_min}min", metadata

    if dataset == "2d_wga_dapi":
        no_csp = any(
            token in upper
            for token in ("NOCSP", "NO_CSP", "WITHOUTCSP", "WITHOUT_CSP")
        )
        csp = ("CSP" in upper) and not no_csp
        metadata["csp"] = csp
        return f"{medium}_{'CSP' if csp else 'noCSP'}", metadata

    if dataset == "3d_mip":
        return medium, metadata

    raise ValueError(f"Unsupported dataset: {dataset}")


def condition_sort_key(dataset: str, condition: str) -> tuple[int, str]:
    """Return condition sort key for the supplied inputs.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        condition (str): Text value specifying condition.

    Returns:
        tuple[int, str]: Collection containing the generated or selected values.

    Example:
        >>> result = condition_sort_key(dataset="2d_time", condition="condition")
    """
    order = DATASET_CONFIGS[dataset].condition_order
    try:
        return order.index(condition), condition
    except ValueError:
        return len(order), condition


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV data from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        list[dict[str, str]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = read_csv(path=Path("path/to/resource"))
    """
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write CSV data to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (Sequence[dict[str, object]]): Text value specifying rows.

    Example:
        >>> write_csv(path=Path("path/to/resource"), rows="rows")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def integer_value(row: dict[str, str], *keys: str) -> int:
    """Return integer value for the supplied inputs.

    Args:
        row (dict[str, str]): Text value specifying row.
        *keys (str): Text value specifying keys.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = integer_value(row="row")
    """
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return 0


def filter_manifest_rows(
    rows: Iterable[dict[str, str]],
    dataset: str,
    include: str = "full",
    excluded_times: set[int] | None = None,
    sample_filters: Sequence[str] | None = None,
    include_failed_pairs: bool = False,
) -> list[dict[str, object]]:
    """Filter manifest rows according to the configured criteria.

    Args:
        rows (Iterable[dict[str, str]]): Text value specifying rows.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        include (str): Text value specifying include. Defaults to ``"full"``.
        excluded_times (set[int] | None): Numerical value controlling excluded times. ``None`` selects the function's default behavior.
        sample_filters (Sequence[str] | None): Text value specifying sample filters. ``None`` selects the function's default behavior.
        include_failed_pairs (bool): Boolean flag controlling whether to failed pairs. Defaults to ``False``.

    Returns:
        list[dict[str, object]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = filter_manifest_rows(rows="rows", dataset="2d_time")
    """
    excluded_times = excluded_times or set()
    selected_samples = {item.strip() for item in sample_filters or () if item.strip()}
    output: list[dict[str, object]] = []

    for source_row in rows:
        row: dict[str, object] = dict(source_row)
        annotation_type = str(source_row.get("annotation_type", "full"))
        split = str(source_row.get("split", "full"))
        if include == "full" and annotation_type != "full":
            continue
        if include in {"train", "validation"} and split != include:
            continue
        if str(source_row.get("overall_status", "PASS")) == "FAIL" and not include_failed_pairs:
            continue

        sample_name = str(source_row.get("sample_name", ""))
        if selected_samples and not any(
            sample_name == sample or sample_name.endswith(sample) for sample in selected_samples
        ):
            continue

        condition, metadata = infer_condition(dataset, sample_name)
        time_value = int(metadata.get("time_min", -1)) if dataset == "2d_time" else -1
        if time_value in excluded_times:
            continue

        row["condition"] = condition
        row.update(metadata)
        row["selection_cell_count"] = integer_value(
            source_row, "n_pca_eligible_instances", "n_instances"
        )
        output.append(row)

    output.sort(
        key=lambda row: (
            condition_sort_key(dataset, str(row.get("condition", ""))),
            str(row.get("sample_name", "")),
            str(row.get("annotation_id", "")),
        )
    )
    return output


def select_one_per_condition(
    rows: Sequence[dict[str, object]],
    dataset: str,
    preferred_min_cells: int,
    max_cells: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select one per condition according to the configured criteria.

    Args:
        rows (Sequence[dict[str, object]]): Text value specifying rows.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        preferred_min_cells (int): Numerical value controlling preferred min cells.
        max_cells (int | None): Maximum permitted value of cells.

    Returns:
        tuple[list[dict[str, object]], list[dict[str, object]]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = select_one_per_condition(
        ...     rows="rows",
        ...     dataset="2d_time",
        ...     preferred_min_cells=1,
        ...     max_cells=1,
        ... )
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("annotation_type", "full")) != "full":
            continue
        n_cells = int(row.get("selection_cell_count", 0) or 0)
        if n_cells <= 0:
            continue
        grouped.setdefault(str(row.get("condition", "UNKNOWN")), []).append(dict(row))

    selected: list[dict[str, object]] = []
    report: list[dict[str, object]] = []
    for condition in DATASET_CONFIGS[dataset].condition_order:
        candidates = grouped.get(condition, [])
        if not candidates:
            continue
        preferred = [
            row
            for row in candidates
            if int(row["selection_cell_count"]) >= preferred_min_cells
            and (max_cells is None or int(row["selection_cell_count"]) <= max_cells)
        ]
        if preferred:
            pool = preferred
            reason = "smallest_within_preferred_range"
        else:
            under_max = [
                row
                for row in candidates
                if max_cells is None or int(row["selection_cell_count"]) <= max_cells
            ]
            pool = under_max or candidates
            reason = "fallback_smallest_available"
        chosen = min(
            pool,
            key=lambda row: (
                int(row["selection_cell_count"]),
                str(row.get("sample_name", "")),
                str(row.get("annotation_id", "")),
            ),
        )
        selected.append(chosen)
        report.append(
            {
                "condition": condition,
                "sample_name": chosen.get("sample_name", ""),
                "annotation_id": chosen.get("annotation_id", ""),
                "n_pca_eligible_cells": chosen.get("selection_cell_count", 0),
                "selection_reason": reason,
                "candidate_count": len(candidates),
            }
        )
    return selected, report


def manifest_path(project_root: Path, dataset: str, source_mode: str) -> Path:
    """Return manifest path for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = manifest_path(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
    return (
        project_root
        / "results"
        / "statistics_preparation"
        / "mask_checks"
        / dataset
        / source_mode
        / "pca_input_manifest.csv"
    )


def mask_summary_path(project_root: Path, dataset: str, source_mode: str) -> Path:
    """Return mask summary path for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = mask_summary_path(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
    return (
        project_root
        / "results"
        / "statistics_preparation"
        / "mask_checks"
        / dataset
        / source_mode
        / "mask_check_summary.csv"
    )


def pca_output_path(
    project_root: Path,
    dataset: str,
    source_mode: str,
    example: bool,
) -> Path:
    """Return principal-component analysis result output path for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        example (bool): Boolean flag controlling example.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = pca_output_path(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     example=True,
        ... )
    """
    stage = "pca_aligned_example" if example else "pca_aligned"
    return project_root / "results" / "statistics_preparation" / stage / dataset / source_mode


def normalization_output_path(
    project_root: Path,
    dataset: str,
    source_mode: str,
    example: bool,
) -> Path:
    """Return normalization output path for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        example (bool): Boolean flag controlling example.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = normalization_output_path(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     example=True,
        ... )
    """
    stage = "normalized_cells_example" if example else "normalized_cells"
    return project_root / "results" / "statistics_preparation" / stage / dataset / source_mode


def graph_output_path(
    project_root: Path,
    dataset: str,
    source_mode: str,
    example: bool,
) -> Path:
    """Return graph output path for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        example (bool): Boolean flag controlling example.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = graph_output_path(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     example=True,
        ... )
    """
    stage = "graphs_example" if example else "graphs"
    return project_root / "results" / "statistics" / stage / dataset / source_mode
