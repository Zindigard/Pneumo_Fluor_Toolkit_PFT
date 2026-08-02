"""Provide command-line and programmatic utilities for results graphs."""


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the Pneumo_Fluor_Toolkit_PFT project root.

    Args:
        start (Optional[Path]): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = find_project_root()
    """
    current = Path.cwd() if start is None else Path(start).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "results").exists():
            return candidate
        if candidate.name == "Pneumo_Fluor_Toolkit_PFT":
            return candidate

    return current


def final_values_dir(project_root: Optional[Path] = None) -> Path:
    """Return final values dir for the supplied inputs.

    Args:
        project_root (Optional[Path]): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = final_values_dir()
    """
    root = find_project_root(project_root)
    return root / "results" / "final_values"


def graphs_dir(project_root: Optional[Path] = None) -> Path:
    """Return graphs dir for the supplied inputs.

    Args:
        project_root (Optional[Path]): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = graphs_dir()
    """
    root = find_project_root(project_root)
    path = root / "results" / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_csv_files(project_root: Optional[Path] = None) -> list[Path]:
    """List CSV data files available in the configured project structure.

    Args:
        project_root (Optional[Path]): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = list_csv_files()
    """
    folder = final_values_dir(project_root)
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.csv"))


COLUMN_ALIASES: dict[str, list[str]] = {
    "condition": ["condition", "group", "media", "medium", "treatment"],
    "time": ["time", "time_min", "timepoint", "time_point", "minutes", "min"],
    "cell_id": ["cell_id", "label", "label_id", "instance_id", "object_id", "cell"],
    "axis": ["axis_pos", "axis_position", "normalized_position", "normalized_axis", "x", "bin_center"],
    "radius": ["radius_norm", "radial_position", "normalized_radius", "r", "radial_bin_center"],
    "value": ["value", "mean_value", "mean_intensity", "intensity", "signal"],
    "cell_count": ["cell_count", "n_cells", "detected_cells", "count"],
    "cell_length": ["cell_length_um", "length_um", "major_axis_length_um", "cell_length", "length"],
    "cell_area": ["cell_area_um2", "area_um2", "cell_area", "area"],
    "cell_volume": ["cell_volume_um3", "volume_um3", "cell_volume", "volume"],
    "homogeneity": [
        "homogeneity",
        "homogeneity_index",
        "boundary_homogeneity",
        "wga_boundary_homogeneity",
        "dapi_homogeneity",
    ],
    "spatial_relation": [
        "wga_dapi_spatial_relation",
        "spatial_relation",
        "spatial_relationship",
        "pearson",
        "pearson_r",
        "correlation",
        "colocalization",
        "colocalisation",
    ],
    "channel": ["channel", "marker", "probe", "pair", "channel_pair"],
    "hada": ["hada", "hada_mean", "hada_intensity", "mean_hada", "hada_norm_intensity"],
    "nada": ["nada", "nada_mean", "nada_intensity", "mean_nada", "nada_norm_intensity"],
    "tada": ["tada", "tada_mean", "tada_intensity", "mean_tada", "tada_norm_intensity"],
    "wga": ["wga", "wga_mean", "wga_intensity", "wga_norm_intensity", "mean_wga"],
    "dapi": ["dapi", "dapi_mean", "dapi_intensity", "dapi_norm_intensity", "mean_dapi"],
}


def normalize_column_name(name: str) -> str:
    """Normalize column name using the configured procedure.

    Args:
        name (str): Name used to identify the current object, resource, or output.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = normalize_column_name(name="name")
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns using the configured procedure.

    Args:
        df (pd.DataFrame): Value specifying df for the operation.

    Returns:
        pd.DataFrame: Result produced by the operation.

    Example:
        >>> result = normalize_columns(df=...)
    """
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def find_column(
    df: pd.DataFrame,
    logical_name: str,
    required: bool = True,
    extra_candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Find column in the available data or project structure.

    Args:
        df (pd.DataFrame): Value specifying df for the operation.
        logical_name (str): Text value specifying logical name.
        required (bool): Boolean flag controlling required. Defaults to ``True``.
        extra_candidates (Optional[Sequence[str]]): Text value specifying extra candidates. ``None`` selects the function's default behavior.

    Returns:
        Optional[str]: Generated or resolved text value.

    Raises:
        KeyError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_column(df=..., logical_name="logical_name")
    """
    cols = set(df.columns)
    candidates: list[str] = []
    if extra_candidates:
        candidates.extend(extra_candidates)
    candidates.extend(COLUMN_ALIASES.get(logical_name, []))
    candidates.append(logical_name)

    for candidate in [normalize_column_name(c) for c in candidates]:
        if candidate in cols:
            return candidate

    if required:
        available = ", ".join(df.columns)
        raise KeyError(
            f"Could not find required column for '{logical_name}'. "
            f"Tried: {candidates}. Available columns: {available}"
        )
    return None


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV data from persistent storage.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.

    Returns:
        pd.DataFrame: Result produced by the operation.

    Example:
        >>> result = load_csv(csv_path=Path("path/to/resource"))
    """
    df = pd.read_csv(csv_path)
    return normalize_columns(df)


@dataclass
class FigureResult:
    """Store validated configuration or result data for figure result."""
    png_path: Path
    pdf_path: Path
    csv_path: Optional[Path] = None
    extra_paths: list[Path] = field(default_factory=list)

    @property
    def all_paths(self) -> list[Path]:
        """Return all paths for the supplied inputs.

        Returns:
            list[Path]: Resolved or generated filesystem path.

        Example:
            >>> instance = FigureResult(...)
            >>> value = instance.all_paths
        """
        paths = [self.png_path, self.pdf_path]
        if self.csv_path is not None:
            paths.append(self.csv_path)
        paths.extend(self.extra_paths)
        return paths


def save_current_figure(output_dir: Path, stem: str, dpi: int = 600) -> FigureResult:
    """Save current figure to persistent storage.

    Args:
        output_dir (Path): Directory where generated resources are written.
        stem (str): Text value specifying stem.
        dpi (int): Resolution of a generated figure in dots per inch. Defaults to ``600``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = save_current_figure(output_dir=Path("path/to/resource"), stem="stem")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return FigureResult(png_path=png_path, pdf_path=pdf_path)


def clean_axis(ax: plt.Axes) -> None:
    """Clean axis before downstream processing.

    Args:
        ax (plt.Axes): Matplotlib axes object on which graphical elements are drawn.

    Example:
        >>> clean_axis(ax=...)
    """
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_summary_csv(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    """Save summary CSV data to persistent storage.

    Args:
        df (pd.DataFrame): Value specifying df for the operation.
        output_dir (Path): Directory where generated resources are written.
        stem (str): Text value specifying stem.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = save_summary_csv(
        ...     df=...,
        ...     output_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def boxplot_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    group_col: Optional[str] = None,
    stem: Optional[str] = None,
    ylabel: Optional[str] = None,
    group_order: Optional[Sequence[str]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> FigureResult:
    """Create a box plot with individual-cell points from a CSV file.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        stem (Optional[str]): Text value specifying stem. ``None`` selects the function's default behavior.
        ylabel (Optional[str]): Text value specifying ylabel. ``None`` selects the function's default behavior.
        group_order (Optional[Sequence[str]]): Text value specifying group order. ``None`` selects the function's default behavior.
        ylim (Optional[tuple[float, float]]): Numerical value controlling ylim. ``None`` selects the function's default behavior.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = boxplot_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    value_col = normalize_column_name(value_col) if value_col else find_column(df, "value")
    group_col = normalize_column_name(group_col) if group_col else find_column(df, "condition")

    plot_df = df[[group_col, value_col]].dropna()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])

    if group_order is None:
        plot_df[group_col] = plot_df[group_col].astype(str)
        preferred = ["THY", "NHS", "THY + CSP", "NHS + CSP"]
        present = list(dict.fromkeys(plot_df[group_col].tolist()))
        group_order = [g for g in preferred if g in present] + [g for g in present if g not in preferred]

    data = [plot_df.loc[plot_df[group_col].astype(str) == str(g), value_col].to_numpy() for g in group_order]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=False)

    rng = np.random.default_rng(42)
    for i, arr in enumerate(data, start=1):
        jitter = rng.uniform(-0.10, 0.10, len(arr))
        ax.scatter(np.full(len(arr), i) + jitter, arr, s=14, alpha=0.35)

    ax.set_xticks(range(1, len(group_order) + 1))
    ax.set_xticklabels(group_order, rotation=15)
    ax.set_ylabel(ylabel or value_col)
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean_axis(ax)
    fig.tight_layout()

    if stem is None:
        stem = f"boxplot_{value_col}"
    result = save_current_figure(output_dir, stem)

    summary = (
        plot_df.groupby(group_col, as_index=False)
        .agg(
            n=(value_col, "size"),
            mean=(value_col, "mean"),
            sd=(value_col, "std"),
            median=(value_col, "median"),
            q1=(value_col, lambda x: x.quantile(0.25)),
            q3=(value_col, lambda x: x.quantile(0.75)),
        )
    )
    result.csv_path = save_summary_csv(summary, output_dir, f"{stem}_summary")
    return result


def time_population_plot_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    stem: str = "time_population_plot",
    ylabel: str = "Detected cells",
) -> FigureResult:
    """Create a time-population line plot. If no value_col is supplied, cells are counted.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        time_col (Optional[str]): Text value specifying time col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"time_population_plot"``.
        ylabel (str): Text value specifying ylabel. Defaults to ``"Detected cells"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = time_population_plot_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    time_col = normalize_column_name(time_col) if time_col else find_column(df, "time")
    group_col = normalize_column_name(group_col) if group_col else find_column(df, "condition", required=False)

    if value_col is not None:
        value_col = normalize_column_name(value_col)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        group_cols = [time_col] + ([group_col] if group_col else [])
        summary = (
            df.groupby(group_cols, as_index=False)
            .agg(mean_value=(value_col, "mean"), sd_value=(value_col, "std"), n=(value_col, "size"))
        )
        y_col = "mean_value"
        yerr_col = "sd_value"
    else:
        cell_col = find_column(df, "cell_id", required=False)
        count_col = find_column(df, "cell_count", required=False)
        group_cols = [time_col] + ([group_col] if group_col else [])
        if count_col:
            df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
            summary = df.groupby(group_cols, as_index=False).agg(count=(count_col, "mean"))
        elif cell_col:
            summary = df.groupby(group_cols, as_index=False).agg(count=(cell_col, "nunique"))
        else:
            summary = df.groupby(group_cols, as_index=False).size().rename(columns={"size": "count"})
        y_col = "count"
        yerr_col = None

    summary[time_col] = pd.to_numeric(summary[time_col], errors="coerce")
    summary = summary.dropna(subset=[time_col]).sort_values(time_col)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if group_col:
        for group, sub in summary.groupby(group_col, sort=False):
            sub = sub.sort_values(time_col)
            if yerr_col and yerr_col in sub.columns:
                ax.errorbar(sub[time_col], sub[y_col], yerr=sub[yerr_col], marker="o", linewidth=1.6, capsize=3, label=str(group))
            else:
                ax.plot(sub[time_col], sub[y_col], marker="o", linewidth=1.6, label=str(group))
        ax.legend(frameon=False)
    else:
        ax.plot(summary[time_col], summary[y_col], marker="o", linewidth=1.6)

    ax.set_xlabel("Time after labelling (min)")
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    fig.tight_layout()

    result = save_current_figure(output_dir, stem)
    result.csv_path = save_summary_csv(summary, output_dir, f"{stem}_summary")
    return result


def profile_from_csv(
    csv_path: Path,
    output_dir: Path,
    coordinate_col: str,
    value_cols: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    stem: str = "profile",
    xlabel: str = "Normalized position",
    ylabel: str = "Normalized intensity",
) -> FigureResult:
    """Create mean line profiles with SEM shading.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        coordinate_col (str): Text value specifying coordinate col.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"profile"``.
        xlabel (str): Text value specifying xlabel. Defaults to ``"Normalized position"``.
        ylabel (str): Text value specifying ylabel. Defaults to ``"Normalized intensity"``.

    Returns:
        FigureResult: Result produced by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = profile_from_csv(
        ...     csv_path=Path("path/to/resource"),
        ...     output_dir=Path("path/to/resource"),
        ...     coordinate_col="coordinate_col",
        ... )
    """
    df = load_csv(csv_path)
    coordinate_col = normalize_column_name(coordinate_col)
    group_col = normalize_column_name(group_col) if group_col else find_column(df, "condition", required=False)

    if value_cols is None:
        possible = []
        for key in ["hada", "nada", "tada", "wga", "dapi", "value"]:
            col = find_column(df, key, required=False)
            if col:
                possible.append(col)
        value_cols = possible

    if not value_cols:
        raise ValueError("No intensity/value columns were found for the profile.")

    value_cols = [normalize_column_name(c) for c in value_cols]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    summary_rows = []

    grouped = df.groupby(group_col, sort=False) if group_col else [("all", df)]

    for group_name, sub in grouped:
        for val_col in value_cols:
            work = sub[[coordinate_col, val_col]].copy()
            work[coordinate_col] = pd.to_numeric(work[coordinate_col], errors="coerce")
            work[val_col] = pd.to_numeric(work[val_col], errors="coerce")
            work = work.dropna()

            summary = (
                work.groupby(coordinate_col, as_index=False)
                .agg(mean_value=(val_col, "mean"), sd_value=(val_col, "std"), n=(val_col, "size"))
                .sort_values(coordinate_col)
            )
            summary["sem_value"] = summary["sd_value"] / np.sqrt(summary["n"].clip(lower=1))

            label = str(group_name) if len(value_cols) == 1 else f"{group_name}: {val_col}"
            ax.plot(summary[coordinate_col], summary["mean_value"], linewidth=1.8, label=label)
            ax.fill_between(
                summary[coordinate_col],
                summary["mean_value"] - summary["sem_value"],
                summary["mean_value"] + summary["sem_value"],
                alpha=0.15,
            )

            summary["group"] = group_name
            summary["value_column"] = val_col
            summary_rows.append(summary)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    clean_axis(ax)
    fig.tight_layout()

    result = save_current_figure(output_dir, stem)
    if summary_rows:
        result.csv_path = save_summary_csv(pd.concat(summary_rows, ignore_index=True), output_dir, f"{stem}_summary")
    return result


def axial_profile_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_cols: Optional[Sequence[str]] = None,
    axis_col: Optional[str] = None,
    group_col: Optional[str] = None,
    stem: str = "axial_profile",
    ylabel: str = "Normalized intensity",
) -> FigureResult:
    """Return axial profile from CSV data for the supplied inputs.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        axis_col (Optional[str]): Text value specifying axis col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"axial_profile"``.
        ylabel (str): Text value specifying ylabel. Defaults to ``"Normalized intensity"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = axial_profile_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    axis_col = normalize_column_name(axis_col) if axis_col else find_column(df, "axis")
    return profile_from_csv(
        csv_path=csv_path,
        output_dir=output_dir,
        coordinate_col=axis_col,
        value_cols=value_cols,
        group_col=group_col,
        stem=stem,
        xlabel="Normalized cell axis",
        ylabel=ylabel,
    )


def radial_profile_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_cols: Optional[Sequence[str]] = None,
    radius_col: Optional[str] = None,
    group_col: Optional[str] = None,
    stem: str = "radial_profile",
    ylabel: str = "Normalized intensity",
) -> FigureResult:
    """Return radial profile from CSV data for the supplied inputs.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        radius_col (Optional[str]): Text value specifying radius col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"radial_profile"``.
        ylabel (str): Text value specifying ylabel. Defaults to ``"Normalized intensity"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = radial_profile_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    radius_col = normalize_column_name(radius_col) if radius_col else find_column(df, "radius")
    return profile_from_csv(
        csv_path=csv_path,
        output_dir=output_dir,
        coordinate_col=radius_col,
        value_cols=value_cols,
        group_col=group_col,
        stem=stem,
        xlabel="Normalized radius (0 = center, 1 = boundary)",
        ylabel=ylabel,
    )


def standardized_cell_map_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    axis_col: Optional[str] = None,
    group_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    stem: str = "standardized_cell_map",
    intensity_label: str = "Normalized intensity",
) -> FigureResult:
    """Create kymograph-like standardized-cell map from long-format CSV.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        axis_col (Optional[str]): Text value specifying axis col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        cell_col (Optional[str]): Text value specifying cell col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"standardized_cell_map"``.
        intensity_label (str): Text value specifying intensity label. Defaults to ``"Normalized intensity"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = standardized_cell_map_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    axis_col = normalize_column_name(axis_col) if axis_col else find_column(df, "axis")
    group_col = normalize_column_name(group_col) if group_col else find_column(df, "condition")
    cell_col = normalize_column_name(cell_col) if cell_col else find_column(df, "cell_id")
    value_col = normalize_column_name(value_col) if value_col else find_column(df, "value")

    df[axis_col] = pd.to_numeric(df[axis_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[axis_col, value_col])

    groups = list(dict.fromkeys(df[group_col].astype(str).tolist()))
    preferred = ["THY", "NHS", "THY + CSP", "NHS + CSP"]
    groups = [g for g in preferred if g in groups] + [g for g in groups if g not in preferred]

    matrices = []
    yticks = []
    ylabels = []
    start = 0

    for group in groups:
        sub = df[df[group_col].astype(str) == group]
        mat = sub.pivot_table(index=cell_col, columns=axis_col, values=value_col, aggfunc="mean")
        mat = mat.reindex(sorted(mat.columns), axis=1)
        arr = mat.to_numpy()

        if arr.shape[1] >= 6:
            pole_bins = max(2, arr.shape[1] // 10)
            pole_score = np.nanmean(arr[:, :pole_bins], axis=1) + np.nanmean(arr[:, -pole_bins:], axis=1)
            sort_idx = np.argsort(-pole_score)
            arr = arr[sort_idx]

        matrices.append(arr)
        yticks.append(start + (arr.shape[0] - 1) / 2)
        ylabels.append(f"{group}\nn={arr.shape[0]}")
        start += arr.shape[0]

    stacked = np.vstack(matrices)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    im = ax.imshow(stacked, aspect="auto", interpolation="nearest", origin="upper", vmin=np.nanmin(stacked), vmax=np.nanmax(stacked))

    start = 0
    for arr in matrices[:-1]:
        start += arr.shape[0]
        ax.axhline(start - 0.5, color="white", linewidth=8)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("Cells grouped by condition")

    n_bins = stacked.shape[1]
    ax.set_xticks([0, (n_bins - 1) / 2, n_bins - 1])
    ax.set_xticklabels(["0\nPole 1", "0.5\nCenter", "1\nPole 2"])
    ax.set_xlabel("Normalized cell axis")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(intensity_label)

    fig.tight_layout()
    return save_current_figure(output_dir, stem)


def colocalization_boxplot_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    group_col: Optional[str] = None,
    pair_col: Optional[str] = None,
    stem: str = "colocalization_boxplot",
) -> FigureResult:
    """Return colocalization boxplot from CSV data for the supplied inputs.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        pair_col (Optional[str]): Text value specifying pair col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"colocalization_boxplot"``.

    Returns:
        FigureResult: Result produced by the operation.

    Raises:
        KeyError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = colocalization_boxplot_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    value_col = normalize_column_name(value_col) if value_col else find_column(df, "spatial_relation")
    pair_col = normalize_column_name(pair_col) if pair_col else find_column(df, "channel", required=False)
    group_col = normalize_column_name(group_col) if group_col else find_column(df, "condition", required=False)
    x_col = pair_col or group_col
    if x_col is None:
        raise KeyError("Need pair/channel or condition/group column.")
    return boxplot_from_csv(csv_path, output_dir, value_col=value_col, group_col=x_col, stem=stem, ylabel="Spatial relationship / correlation")


def correlation_plot_from_csv(
    csv_path: Path,
    output_dir: Path,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    stem: str = "correlation_plot",
) -> FigureResult:
    """Return correlation plot from CSV data for the supplied inputs.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        x_col (str): Text value specifying x col.
        y_col (str): Text value specifying y col.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"correlation_plot"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = correlation_plot_from_csv(
        ...     csv_path=Path("path/to/resource"),
        ...     output_dir=Path("path/to/resource"),
        ...     x_col="x_col",
        ...     y_col="y_col",
        ... )
    """
    df = load_csv(csv_path)
    x_col = normalize_column_name(x_col)
    y_col = normalize_column_name(y_col)
    group_col = normalize_column_name(group_col) if group_col else find_column(df, "condition", required=False)

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    summary_rows = []

    if group_col and group_col in df.columns:
        for group, sub in df.groupby(group_col, sort=False):
            ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.45, label=str(group))
            r = sub[[x_col, y_col]].corr().iloc[0, 1] if len(sub) > 2 else np.nan
            summary_rows.append({"group": group, "n": len(sub), "pearson_r": r})
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(df[x_col], df[y_col], s=18, alpha=0.45)
        r = df[[x_col, y_col]].corr().iloc[0, 1] if len(df) > 2 else np.nan
        summary_rows.append({"group": "all", "n": len(df), "pearson_r": r})

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    clean_axis(ax)
    fig.tight_layout()

    result = save_current_figure(output_dir, stem)
    result.csv_path = save_summary_csv(pd.DataFrame(summary_rows), output_dir, f"{stem}_summary")
    return result




# -----------------------------------------------------------------------------
# 3D SIM graph helpers
# -----------------------------------------------------------------------------
# These functions implement the final 3D figure logic discussed for the thesis:
#   - blue and green are switched relative to the earlier intermediate version;
#   - blue has a central axial signal and remains non-zero at poles in kymographs;
#   - green is the most homogeneous and is higher at the poles;
#   - red is the least homogeneous and remains lower than blue in the homogeneity plot.
#
# The functions still use the same data-loading pattern as the 2D functions:
# values are read from CSV files in results/final_values, then summarized and plotted.
# No synthetic values are generated inside these functions.

CHANNEL_COLOURS_3D: dict[str, str] = {
    "blue": "tab:blue",
    "green": "tab:green",
    "red": "tab:red",
}

CHANNEL_CMAPS_3D: dict[str, str] = {
    "blue": "Blues",
    "green": "Greens",
    "red": "Reds",
}

CHANNEL_ORDER_3D_FINAL: list[str] = ["green", "blue", "red"]
CONDITION_ORDER_3D: list[str] = ["THY", "NHS"]


def _ordered_present(values: Sequence[object], preferred: Sequence[str]) -> list[str]:
    """Return ordered present for the supplied inputs.

    Args:
        values (Sequence[object]): Value specifying values for the operation.
        preferred (Sequence[str]): Text value specifying preferred.

    Returns:
        list[str]: Collection containing the generated or selected values.

    Example:
        >>> result = _ordered_present(values=[], preferred="preferred")
    """
    present = list(dict.fromkeys(str(v) for v in values if pd.notna(v)))
    upper_map = {p.upper(): p for p in present}
    ordered: list[str] = []
    for item in preferred:
        if item.upper() in upper_map:
            ordered.append(upper_map[item.upper()])
    ordered.extend([p for p in present if p not in ordered])
    return ordered


def _is_numeric_series(series: pd.Series) -> bool:
    """Determine whether numeric series satisfies the stated condition.

    Args:
        series (pd.Series): Value specifying series for the operation.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _is_numeric_series(series=...)
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() > 0.5


def _classify_3d_source_channel(raw: object) -> str:
    """Classify CSV channel names before the final blue/green switch.

    Args:
        raw (object): Value specifying raw for the operation.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _classify_3d_source_channel(raw=...)
    """
    name = normalize_column_name(str(raw))
    if any(key in name for key in ["hada", "blue", "b_channel", "channel_b"]):
        return "old_blue"
    if any(key in name for key in ["nada", "green", "g_channel", "channel_g"]):
        return "old_green"
    if any(key in name for key in ["tada", "red", "r_channel", "channel_r"]):
        return "red"
    return name


def map_3d_channel(raw: object, switch_blue_green: bool = True) -> str:
    """Map HADA/NADA/TADA or blue/green/red labels to final thesis channel names.

    With switch_blue_green=True, the earlier blue-like channel is plotted as green,
    and the earlier green-like channel is plotted as blue. This is the final setting
    used for the discussed 3D graphs.

    Args:
        raw (object): Value specifying raw for the operation.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = map_3d_channel(raw=...)
    """
    source = _classify_3d_source_channel(raw)
    if switch_blue_green:
        if source == "old_blue":
            return "green"
        if source == "old_green":
            return "blue"
    else:
        if source == "old_blue":
            return "blue"
        if source == "old_green":
            return "green"
    if source == "red":
        return "red"
    return source


def _discover_3d_value_columns(df: pd.DataFrame, value_cols: Optional[Sequence[str]] = None) -> list[str]:
    """Discover three-dimensional data value columns in the configured project structure.

    Args:
        df (pd.DataFrame): Value specifying df for the operation.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.

    Returns:
        list[str]: Collection containing the generated or selected values.

    Example:
        >>> result = _discover_3d_value_columns(df=...)
    """
    if value_cols:
        return [normalize_column_name(c) for c in value_cols]

    discovered: list[str] = []
    for logical in ["hada", "nada", "tada"]:
        col = find_column(df, logical, required=False)
        if col and col not in discovered and _is_numeric_series(df[col]):
            discovered.append(col)

    for col in df.columns:
        if col in discovered:
            continue
        name = normalize_column_name(col)
        if any(key in name for key in ["hada", "nada", "tada", "blue", "green", "red"]):
            if _is_numeric_series(df[col]):
                discovered.append(col)

    return discovered


def _three_d_long_values(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    channel_col: Optional[str] = None,
    switch_blue_green: bool = True,
    logical_value: str = "value",
    id_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return long-format columns: id columns + condition/channel/value.

    Accepts either:
      1) long CSV: condition, channel, value;
      2) wide CSV: condition and one column per channel, e.g. HADA/NADA/TADA,
         blue/green/red, or *_intensity / *_homogeneity variants.

    Args:
        df (pd.DataFrame): Value specifying df for the operation.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        channel_col (Optional[str]): Text value specifying channel col. ``None`` selects the function's default behavior.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.
        logical_value (str): Text value specifying logical value. Defaults to ``"value"``.
        id_cols (Optional[Sequence[str]]): Text value specifying id cols. ``None`` selects the function's default behavior.

    Returns:
        pd.DataFrame: Result produced by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _three_d_long_values(df=...)
    """
    work = df.copy()

    condition_col = find_column(work, "condition", required=False)
    if condition_col is None:
        work["condition"] = "all"
        condition_col = "condition"

    channel_col = normalize_column_name(channel_col) if channel_col else find_column(work, "channel", required=False)

    if value_col:
        value_col = normalize_column_name(value_col)
    else:
        value_col = find_column(work, logical_value, required=False)
        if value_col is None and logical_value != "value":
            value_col = find_column(work, "value", required=False)

    id_cols_norm = [normalize_column_name(c) for c in (id_cols or []) if c]
    id_cols_norm = [c for c in id_cols_norm if c in work.columns]

    if channel_col is not None and value_col is not None:
        keep = list(dict.fromkeys([*id_cols_norm, condition_col, channel_col, value_col]))
        long = work[keep].copy()
        long = long.rename(columns={channel_col: "source_channel", value_col: "value"})
    else:
        wide_cols = _discover_3d_value_columns(work, value_cols=value_cols)
        if not wide_cols:
            available = ", ".join(work.columns)
            raise ValueError(
                "Could not detect 3D channel columns. Provide --value-cols or use a long CSV "
                f"with channel and value columns. Available columns: {available}"
            )
        keep_ids = list(dict.fromkeys([*id_cols_norm, condition_col]))
        long = work.melt(
            id_vars=keep_ids,
            value_vars=wide_cols,
            var_name="source_channel",
            value_name="value",
        )

    long["condition"] = long[condition_col].astype(str)
    long["channel"] = long["source_channel"].map(lambda v: map_3d_channel(v, switch_blue_green=switch_blue_green))
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long[long["channel"].isin(["blue", "green", "red"])]
    return long


def _save_figure_pair(output_dir: Path, stem: str, dpi: int = 600) -> tuple[Path, Path]:
    """Save figure pair to persistent storage.

    Args:
        output_dir (Path): Directory where generated resources are written.
        stem (str): Text value specifying stem.
        dpi (int): Resolution of a generated figure in dots per inch. Defaults to ``600``.

    Returns:
        tuple[Path, Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _save_figure_pair(output_dir=Path("path/to/resource"), stem="stem")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def _line_style_for_condition(condition: str) -> str:
    """Return line style for condition for the supplied inputs.

    Args:
        condition (str): Text value specifying condition.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _line_style_for_condition(condition="condition")
    """
    return "--" if condition.upper() == "NHS" else "-"


def _summarize_profile(long: pd.DataFrame, coordinate_col: str) -> pd.DataFrame:
    """Summarize profile for reporting.

    Args:
        long (pd.DataFrame): Value specifying long for the operation.
        coordinate_col (str): Text value specifying coordinate col.

    Returns:
        pd.DataFrame: Result produced by the operation.

    Example:
        >>> result = _summarize_profile(long=..., coordinate_col="coordinate_col")
    """
    work = long.copy()
    work[coordinate_col] = pd.to_numeric(work[coordinate_col], errors="coerce")
    work = work.dropna(subset=[coordinate_col, "value"])
    summary = (
        work.groupby(["condition", "channel", coordinate_col], as_index=False)
        .agg(mean_value=("value", "mean"), sd_value=("value", "std"), n=("value", "size"))
        .sort_values(["condition", "channel", coordinate_col])
    )
    summary["sem_value"] = summary["sd_value"] / np.sqrt(summary["n"].clip(lower=1))
    return summary


def three_d_axial_profile_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    axis_col: Optional[str] = None,
    group_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    stem: str = "3d_sim_mip_axial_profiles",
    switch_blue_green: bool = True,
    ylabel: str = "Normalized fluorescence intensity",
) -> FigureResult:
    """Create the final 3D SIM axial profile graph from CSV values.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        axis_col (Optional[str]): Text value specifying axis col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        channel_col (Optional[str]): Text value specifying channel col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"3d_sim_mip_axial_profiles"``.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.
        ylabel (str): Text value specifying ylabel. Defaults to ``"Normalized fluorescence intensity"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = three_d_axial_profile_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    axis_col = normalize_column_name(axis_col) if axis_col else find_column(df, "axis")
    if group_col:
        group_col = normalize_column_name(group_col)
        df = df.rename(columns={group_col: "condition"})

    long = _three_d_long_values(
        df,
        value_col=value_col,
        value_cols=value_cols,
        channel_col=channel_col,
        switch_blue_green=switch_blue_green,
        id_cols=[axis_col],
    )
    long[axis_col] = pd.to_numeric(long[axis_col], errors="coerce")
    summary = _summarize_profile(long, axis_col)

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=180)
    conditions = _ordered_present(summary["condition"].unique(), CONDITION_ORDER_3D)
    for channel in ["blue", "red", "green"]:
        for condition in conditions:
            sub = summary[(summary["channel"] == channel) & (summary["condition"] == condition)]
            if sub.empty:
                continue
            ax.plot(
                sub[axis_col],
                sub["mean_value"],
                color=CHANNEL_COLOURS_3D[channel],
                linestyle=_line_style_for_condition(condition),
                linewidth=2.0,
                label=f"{condition} {channel}",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.97)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Pole 1", "Midcell", "Pole 2"])
    ax.set_xlabel("Normalized cell axis")
    ax.set_ylabel(ylabel)
    ax.set_title("3D SIM MIP axial fluorescence profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, ncol=2, loc="upper right")
    fig.tight_layout()

    png_path, pdf_path = _save_figure_pair(output_dir, stem)
    summary_csv = save_summary_csv(summary, output_dir, f"{stem}_summary")
    return FigureResult(png_path=png_path, pdf_path=pdf_path, csv_path=summary_csv)


def _plot_three_d_radial_pair(
    summary: pd.DataFrame,
    radius_col: str,
    output_dir: Path,
    channels: tuple[str, str],
    stem: str,
) -> FigureResult:
    """Plot three d radial pair for visual assessment.

    Args:
        summary (pd.DataFrame): Value specifying summary for the operation.
        radius_col (str): Text value specifying radius col.
        output_dir (Path): Directory where generated resources are written.
        channels (tuple[str, str]): Channel indices or identifiers selected for processing.
        stem (str): Text value specifying stem.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = _plot_three_d_radial_pair(
        ...     summary=...,
        ...     radius_col="radius_col",
        ...     output_dir=Path("path/to/resource"),
        ...     channels="channels",
        ...     stem="stem",
        ... )
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    conditions = _ordered_present(summary["condition"].unique(), CONDITION_ORDER_3D)
    for channel in channels:
        for condition in conditions:
            sub = summary[(summary["channel"] == channel) & (summary["condition"] == condition)]
            if sub.empty:
                continue
            ax.plot(
                sub[radius_col],
                sub["mean_value"],
                color=CHANNEL_COLOURS_3D[channel],
                linestyle=_line_style_for_condition(condition),
                linewidth=2.0,
                label=f"{condition} {channel}",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.97)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Cell centre", "Intermediate radius", "Cell boundary"])
    ax.set_xlabel("Normalized radial distance")
    ax.set_ylabel("Normalized fluorescence intensity")
    ax.set_title(f"3D SIM MIP radial profiles: {channels[0]} vs {channels[1]}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, ncol=2, loc="upper right")
    fig.tight_layout()
    png_path, pdf_path = _save_figure_pair(output_dir, stem)
    return FigureResult(png_path=png_path, pdf_path=pdf_path)


def three_d_radial_profiles_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    radius_col: Optional[str] = None,
    group_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    stem: str = "3d_sim_mip_radial_profiles",
    switch_blue_green: bool = True,
) -> FigureResult:
    """Create the three final 3D radial pairwise profiles from CSV values.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        radius_col (Optional[str]): Text value specifying radius col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        channel_col (Optional[str]): Text value specifying channel col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"3d_sim_mip_radial_profiles"``.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = three_d_radial_profiles_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    radius_col = normalize_column_name(radius_col) if radius_col else find_column(df, "radius")
    if group_col:
        group_col = normalize_column_name(group_col)
        df = df.rename(columns={group_col: "condition"})

    long = _three_d_long_values(
        df,
        value_col=value_col,
        value_cols=value_cols,
        channel_col=channel_col,
        switch_blue_green=switch_blue_green,
        id_cols=[radius_col],
    )
    long[radius_col] = pd.to_numeric(long[radius_col], errors="coerce")
    summary = _summarize_profile(long, radius_col)
    summary_csv = save_summary_csv(summary, output_dir, f"{stem}_summary")

    pairs = [("red", "green"), ("red", "blue"), ("blue", "green")]
    results: list[FigureResult] = []
    for pair in pairs:
        pair_stem = f"{stem}_{pair[0]}_vs_{pair[1]}"
        results.append(_plot_three_d_radial_pair(summary, radius_col, output_dir, pair, pair_stem))

    primary = results[0]
    primary.csv_path = summary_csv
    for extra in results[1:]:
        primary.extra_paths.extend([extra.png_path, extra.pdf_path])
    return primary


def three_d_homogeneity_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    stem: str = "3d_sim_mip_channel_homogeneity",
    switch_blue_green: bool = True,
) -> FigureResult:
    """Create final 3D channel homogeneity box plot.

    Final order is Green > Blue > Red. The plot is data-driven: if the CSV values
    differ from this pattern, the actual values are still shown.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        channel_col (Optional[str]): Text value specifying channel col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"3d_sim_mip_channel_homogeneity"``.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = three_d_homogeneity_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    if group_col:
        group_col = normalize_column_name(group_col)
        df = df.rename(columns={group_col: "condition"})

    long = _three_d_long_values(
        df,
        value_col=value_col,
        value_cols=value_cols,
        channel_col=channel_col,
        switch_blue_green=switch_blue_green,
        logical_value="homogeneity",
    )

    conditions = _ordered_present(long["condition"].unique(), CONDITION_ORDER_3D)
    channels = ["green", "blue", "red"]

    data: list[np.ndarray] = []
    positions: list[int] = []
    labels: list[str] = []
    colors: list[str] = []
    pos = 1
    for channel in channels:
        for condition in conditions:
            arr = long[(long["channel"] == channel) & (long["condition"] == condition)]["value"].to_numpy()
            if arr.size == 0:
                continue
            data.append(arr)
            positions.append(pos)
            labels.append(condition)
            colors.append(CHANNEL_COLOURS_3D[channel])
            pos += 1
        pos += 1

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.7, "color": "black"},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    rng = np.random.default_rng(42)
    for position, arr, color in zip(positions, data, colors):
        jitter = rng.normal(0, 0.055, size=len(arr))
        ax.scatter(np.full(len(arr), position) + jitter, arr, s=13, alpha=0.60, color=color, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    # channel labels centered below each condition pair
    group_centres: list[tuple[float, str]] = []
    for channel in channels:
        channel_positions = [p for p, c in zip(positions, colors) if c == CHANNEL_COLOURS_3D[channel]]
        if channel_positions:
            group_centres.append((float(np.mean(channel_positions)), channel.capitalize()))
    for centre, label in group_centres:
        ax.text(centre, -0.105, label, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=11)

    ax.set_xlim(min(positions) - 0.75, max(positions) + 0.75)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Spatial homogeneity")
    ax.set_title("3D SIM MIP channel homogeneity")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    png_path, pdf_path = _save_figure_pair(output_dir, stem)
    summary = (
        long.groupby(["channel", "condition"], as_index=False)
        .agg(n=("value", "size"), mean=("value", "mean"), sd=("value", "std"), median=("value", "median"))
    )
    summary_csv = save_summary_csv(summary, output_dir, f"{stem}_summary")
    return FigureResult(png_path=png_path, pdf_path=pdf_path, csv_path=summary_csv)


def _plot_single_three_d_kymograph(
    long: pd.DataFrame,
    axis_col: str,
    cell_col: str,
    output_dir: Path,
    channel: str,
    stem: str,
    intensity_label: str,
) -> FigureResult:
    """Plot single three d kymograph for visual assessment.

    Args:
        long (pd.DataFrame): Value specifying long for the operation.
        axis_col (str): Text value specifying axis col.
        cell_col (str): Text value specifying cell col.
        output_dir (Path): Directory where generated resources are written.
        channel (str): Channel index or channel identifier selected for processing.
        stem (str): Text value specifying stem.
        intensity_label (str): Text value specifying intensity label.

    Returns:
        FigureResult: Result produced by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _plot_single_three_d_kymograph(
        ...     long=...,
        ...     axis_col="axis_col",
        ...     cell_col="cell_col",
        ...     output_dir=Path("path/to/resource"),
        ...     channel="channel",
        ...     stem="stem",
        ...     intensity_label="intensity_label",
        ... )
    """
    channel_df = long[long["channel"] == channel].copy()
    if channel_df.empty:
        raise ValueError(f"No rows found for 3D channel '{channel}'.")

    conditions = _ordered_present(channel_df["condition"].unique(), CONDITION_ORDER_3D)
    matrices: list[np.ndarray] = []
    xticks: list[float] = []
    xlabels: list[str] = []
    start = 0

    for condition in conditions:
        sub = channel_df[channel_df["condition"] == condition]
        mat = sub.pivot_table(index=axis_col, columns=cell_col, values="value", aggfunc="mean")
        mat = mat.reindex(sorted(mat.index), axis=0)
        mat = mat.reindex(sorted(mat.columns), axis=1)
        arr = mat.to_numpy(dtype=float)
        if arr.size == 0:
            continue
        # Sort cells by central-band intensity for a readable pooled kymograph.
        axis_values = np.asarray(mat.index, dtype=float)
        centre_mask = (axis_values > 0.38) & (axis_values < 0.62)
        if centre_mask.any():
            order = np.argsort(np.nanmean(arr[centre_mask, :], axis=0))
            arr = arr[:, order]
        matrices.append(arr)
        xticks.append(start + arr.shape[1] / 2)
        xlabels.append(f"{condition}\nn={arr.shape[1]}")
        start += arr.shape[1]

    if not matrices:
        raise ValueError(f"No matrix could be built for 3D channel '{channel}'.")

    stacked = np.concatenate(matrices, axis=1)

    fig, ax = plt.subplots(figsize=(13.5, 4.6), dpi=180)
    im = ax.imshow(
        stacked,
        aspect="auto",
        origin="lower",
        cmap=CHANNEL_CMAPS_3D[channel],
        vmin=0,
        vmax=0.95,
        extent=[0, stacked.shape[1], 0, 1],
        interpolation="nearest",
    )

    start = 0
    for arr in matrices[:-1]:
        start += arr.shape[1]
        ax.axvline(start, color="black", linewidth=1.0)

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Pole 1", "0.5", "Pole 2"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Cells grouped by condition")
    ax.set_ylabel("Normalized cell axis")
    ax.set_title(f"3D SIM MIP {channel} channel kymograph")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(intensity_label)
    fig.tight_layout()

    png_path, pdf_path = _save_figure_pair(output_dir, stem)
    return FigureResult(png_path=png_path, pdf_path=pdf_path)


def three_d_kymographs_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    axis_col: Optional[str] = None,
    group_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    stem: str = "3d_sim_mip_kymograph",
    intensity_label: str = "Normalized fluorescence intensity",
    switch_blue_green: bool = True,
) -> FigureResult:
    """Create final 3D blue, green, and red kymographs from long/wide CSV values.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        axis_col (Optional[str]): Text value specifying axis col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        channel_col (Optional[str]): Text value specifying channel col. ``None`` selects the function's default behavior.
        cell_col (Optional[str]): Text value specifying cell col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"3d_sim_mip_kymograph"``.
        intensity_label (str): Text value specifying intensity label. Defaults to ``"Normalized fluorescence intensity"``.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = three_d_kymographs_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    df = load_csv(csv_path)
    axis_col = normalize_column_name(axis_col) if axis_col else find_column(df, "axis")
    cell_col = normalize_column_name(cell_col) if cell_col else find_column(df, "cell_id")
    if group_col:
        group_col = normalize_column_name(group_col)
        df = df.rename(columns={group_col: "condition"})

    long = _three_d_long_values(
        df,
        value_col=value_col,
        value_cols=value_cols,
        channel_col=channel_col,
        switch_blue_green=switch_blue_green,
        id_cols=[axis_col, cell_col],
    )
    long[axis_col] = pd.to_numeric(long[axis_col], errors="coerce")
    long = long.dropna(subset=[axis_col, "value"])

    results: list[FigureResult] = []
    for channel in ["blue", "green", "red"]:
        results.append(
            _plot_single_three_d_kymograph(
                long=long,
                axis_col=axis_col,
                cell_col=cell_col,
                output_dir=output_dir,
                channel=channel,
                stem=f"{stem}_{channel}",
                intensity_label=intensity_label,
            )
        )

    summary = (
        long.groupby(["channel", "condition", axis_col], as_index=False)
        .agg(mean_value=("value", "mean"), sd_value=("value", "std"), n=("value", "size"))
    )
    summary_csv = save_summary_csv(summary, output_dir, f"{stem}_summary")

    primary = results[0]
    primary.csv_path = summary_csv
    for extra in results[1:]:
        primary.extra_paths.extend([extra.png_path, extra.pdf_path])
    return primary



def three_d_all_graphs_from_csv(
    csv_path: Path,
    output_dir: Path,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    axis_col: Optional[str] = None,
    radius_col: Optional[str] = None,
    group_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    stem: str = "3d_sim_mip",
    intensity_label: str = "Normalized fluorescence intensity",
    switch_blue_green: bool = True,
    ylabel: str = "Normalized fluorescence intensity",
) -> FigureResult:
    """Create the full discussed 3D graph set from one combined CSV file.

    The CSV must contain the columns required by each graph type. In practice this
    works best with one long table containing condition, channel, cell_id, axis,
    radius, value, and optionally homogeneity.

    Args:
        csv_path (Path): Filesystem path associated with CSV data.
        output_dir (Path): Directory where generated resources are written.
        value_col (Optional[str]): Text value specifying value col. ``None`` selects the function's default behavior.
        value_cols (Optional[Sequence[str]]): Text value specifying value cols. ``None`` selects the function's default behavior.
        axis_col (Optional[str]): Text value specifying axis col. ``None`` selects the function's default behavior.
        radius_col (Optional[str]): Text value specifying radius col. ``None`` selects the function's default behavior.
        group_col (Optional[str]): Text value specifying group col. ``None`` selects the function's default behavior.
        channel_col (Optional[str]): Text value specifying channel col. ``None`` selects the function's default behavior.
        cell_col (Optional[str]): Text value specifying cell col. ``None`` selects the function's default behavior.
        stem (str): Text value specifying stem. Defaults to ``"3d_sim_mip"``.
        intensity_label (str): Text value specifying intensity label. Defaults to ``"Normalized fluorescence intensity"``.
        switch_blue_green (bool): Boolean flag controlling switch blue green. Defaults to ``True``.
        ylabel (str): Text value specifying ylabel. Defaults to ``"Normalized fluorescence intensity"``.

    Returns:
        FigureResult: Result produced by the operation.

    Example:
        >>> result = three_d_all_graphs_from_csv(csv_path=Path("path/to/resource"), output_dir=Path("path/to/resource"))
    """
    results: list[FigureResult] = []
    results.append(
        three_d_axial_profile_from_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            value_col=value_col,
            value_cols=value_cols,
            axis_col=axis_col,
            group_col=group_col,
            channel_col=channel_col,
            stem=f"{stem}_axial_profiles",
            switch_blue_green=switch_blue_green,
            ylabel=ylabel,
        )
    )
    results.append(
        three_d_radial_profiles_from_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            value_col=value_col,
            value_cols=value_cols,
            radius_col=radius_col,
            group_col=group_col,
            channel_col=channel_col,
            stem=f"{stem}_radial_profiles",
            switch_blue_green=switch_blue_green,
        )
    )
    try:
        results.append(
            three_d_kymographs_from_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                value_col=value_col,
                value_cols=value_cols,
                axis_col=axis_col,
                group_col=group_col,
                channel_col=channel_col,
                cell_col=cell_col,
                stem=f"{stem}_kymograph",
                intensity_label=intensity_label,
                switch_blue_green=switch_blue_green,
            )
        )
    except KeyError:
        # A profile-only CSV may not contain cell_id. Axial/radial graphs still remain valid.
        pass
    try:
        results.append(
            three_d_homogeneity_from_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                value_col=None,
                value_cols=None,
                group_col=group_col,
                channel_col=channel_col,
                stem=f"{stem}_channel_homogeneity",
                switch_blue_green=switch_blue_green,
            )
        )
    except (KeyError, ValueError):
        pass

    primary = results[0]
    for extra in results[1:]:
        primary.extra_paths.extend([extra.png_path, extra.pdf_path])
        if extra.csv_path is not None:
            primary.extra_paths.append(extra.csv_path)
        primary.extra_paths.extend(extra.extra_paths)
    return primary


def run_graph(
    dataset: str,
    graph: str,
    csv_path: Path,
    project_root: Optional[Path] = None,
    output_subdir: Optional[str] = None,
    **kwargs,
) -> FigureResult:
    """Dispatch graph creation based on dataset and graph name.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        graph (str): Text value specifying graph.
        csv_path (Path): Filesystem path associated with CSV data.
        project_root (Optional[Path]): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.
        output_subdir (Optional[str]): Text value specifying output subdir. ``None`` selects the function's default behavior.
        **kwargs (Any): Additional keyword arguments forwarded to the underlying callable.

    Returns:
        FigureResult: Result produced by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = run_graph(
        ...     dataset="2d_time",
        ...     graph="graph",
        ...     csv_path=Path("path/to/resource"),
        ... )
    """
    dataset = dataset.lower().strip()
    graph = graph.lower().strip()
    out = graphs_dir(project_root) / (output_subdir or dataset)
    out.mkdir(parents=True, exist_ok=True)

    is_3d = dataset in {"3d_sim", "3d", "sim", "3d_sim_mip"}

    if is_3d and graph in {"axial_profile", "axial", "3d_axial"}:
        return three_d_axial_profile_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            value_cols=kwargs.get("value_cols"),
            axis_col=kwargs.get("axis_col"),
            group_col=kwargs.get("group_col"),
            channel_col=kwargs.get("channel_col"),
            stem=(kwargs.get("stem") or "3d_sim_mip_axial_profiles"),
            switch_blue_green=kwargs.get("switch_blue_green", True),
            ylabel=(kwargs.get("ylabel") or "Normalized fluorescence intensity"),
        )

    if is_3d and graph in {"radial_profile", "radial", "3d_radial"}:
        return three_d_radial_profiles_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            value_cols=kwargs.get("value_cols"),
            radius_col=kwargs.get("radius_col"),
            group_col=kwargs.get("group_col"),
            channel_col=kwargs.get("channel_col"),
            stem=(kwargs.get("stem") or "3d_sim_mip_radial_profiles"),
            switch_blue_green=kwargs.get("switch_blue_green", True),
        )

    if is_3d and graph in {"kymograph", "standardized_map", "standardized_cell_map", "3d_kymograph"}:
        return three_d_kymographs_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            value_cols=kwargs.get("value_cols"),
            axis_col=kwargs.get("axis_col"),
            group_col=kwargs.get("group_col"),
            channel_col=kwargs.get("channel_col"),
            cell_col=kwargs.get("cell_col"),
            stem=(kwargs.get("stem") or "3d_sim_mip_kymograph"),
            intensity_label=(kwargs.get("intensity_label") or "Normalized fluorescence intensity"),
            switch_blue_green=kwargs.get("switch_blue_green", True),
        )

    if is_3d and graph in {"homogeneity", "channel_homogeneity", "3d_homogeneity"}:
        return three_d_homogeneity_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            value_cols=kwargs.get("value_cols"),
            group_col=kwargs.get("group_col"),
            channel_col=kwargs.get("channel_col"),
            stem=(kwargs.get("stem") or "3d_sim_mip_channel_homogeneity"),
            switch_blue_green=kwargs.get("switch_blue_green", True),
        )

    if is_3d and graph in {"all", "all_3d", "3d_all", "summary_3d"}:
        return three_d_all_graphs_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            value_cols=kwargs.get("value_cols"),
            axis_col=kwargs.get("axis_col"),
            radius_col=kwargs.get("radius_col"),
            group_col=kwargs.get("group_col"),
            channel_col=kwargs.get("channel_col"),
            cell_col=kwargs.get("cell_col"),
            stem=(kwargs.get("stem") or "3d_sim_mip"),
            intensity_label=(kwargs.get("intensity_label") or "Normalized fluorescence intensity"),
            switch_blue_green=kwargs.get("switch_blue_green", True),
        )

    if graph in {"cell_count", "detected_cells", "time_population"}:
        return time_population_plot_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            time_col=kwargs.get("time_col"),
            group_col=kwargs.get("group_col"),
            stem=(kwargs.get("stem") or f"{dataset}_{graph}"),
            ylabel=(kwargs.get("ylabel") or "Detected cells"),
        )

    if graph in {"cell_length", "length"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col", "cell_length_um"), group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_cell_length"), ylabel=(kwargs.get("ylabel") or "Cell length (µm)"))

    if graph in {"cell_area", "area"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col", "cell_area_um2"), group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_cell_area"), ylabel=(kwargs.get("ylabel") or "Cell area (µm²)"))

    if graph in {"cell_volume", "volume"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col", "cell_volume_um3"), group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_cell_volume"), ylabel=(kwargs.get("ylabel") or "Cell volume (µm³)"))

    if graph in {"homogeneity", "wga_homogeneity", "dapi_homogeneity"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col"), group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_{graph}"), ylabel=(kwargs.get("ylabel") or graph.replace("_", " ").title()))

    if graph in {"axial_profile", "wga_axial", "dapi_axial", "hada_axial"}:
        return axial_profile_from_csv(csv_path, out, value_cols=kwargs.get("value_cols"), axis_col=kwargs.get("axis_col"), group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_{graph}"), ylabel=(kwargs.get("ylabel") or "Normalized intensity"))

    if graph in {"radial_profile", "radial"}:
        return radial_profile_from_csv(csv_path, out, value_cols=kwargs.get("value_cols"), radius_col=kwargs.get("radius_col"), group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_radial_profile"), ylabel=(kwargs.get("ylabel") or "Normalized intensity"))

    if graph in {"kymograph", "standardized_map", "standardized_cell_map"}:
        return standardized_cell_map_from_csv(csv_path, out, value_col=kwargs.get("value_col"), axis_col=kwargs.get("axis_col"), group_col=kwargs.get("group_col"), cell_col=kwargs.get("cell_col"), stem=(kwargs.get("stem") or f"{dataset}_{graph}"), intensity_label=kwargs.get("intensity_label", "Normalized intensity"))

    if graph in {"colocalization", "spatial_relationship", "spatial_relation"}:
        return colocalization_boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col"), group_col=kwargs.get("group_col"), pair_col=kwargs.get("pair_col"), stem=(kwargs.get("stem") or f"{dataset}_{graph}"))

    if graph in {"correlation", "scatter"}:
        if not kwargs.get("x_col") or not kwargs.get("y_col"):
            raise ValueError("Correlation plot requires x_col and y_col.")
        return correlation_plot_from_csv(csv_path, out, x_col=kwargs["x_col"], y_col=kwargs["y_col"], group_col=kwargs.get("group_col"), stem=(kwargs.get("stem") or f"{dataset}_correlation"))

    raise ValueError(f"Unknown graph type: {graph}")
