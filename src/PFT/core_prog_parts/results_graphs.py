
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the Pneumo_Fluor_Toolkit_PFT project root."""
    current = Path.cwd() if start is None else Path(start).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "results").exists():
            return candidate
        if candidate.name == "Pneumo_Fluor_Toolkit_PFT":
            return candidate

    return current


def final_values_dir(project_root: Optional[Path] = None) -> Path:
    root = find_project_root(project_root)
    return root / "results" / "final_values"


def graphs_dir(project_root: Optional[Path] = None) -> Path:
    root = find_project_root(project_root)
    path = root / "results" / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_csv_files(project_root: Optional[Path] = None) -> list[Path]:
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
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def find_column(
    df: pd.DataFrame,
    logical_name: str,
    required: bool = True,
    extra_candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
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
    df = pd.read_csv(csv_path)
    return normalize_columns(df)


@dataclass
class FigureResult:
    png_path: Path
    pdf_path: Path
    csv_path: Optional[Path] = None


def save_current_figure(output_dir: Path, stem: str, dpi: int = 600) -> FigureResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return FigureResult(png_path=png_path, pdf_path=pdf_path)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_summary_csv(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
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
    """Create a box plot with individual-cell points from a CSV file."""
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
    """Create a time-population line plot. If no value_col is supplied, cells are counted."""
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
    """Create mean line profiles with SEM shading."""
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
    """Create kymograph-like standardized-cell map from long-format CSV."""
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


def run_graph(
    dataset: str,
    graph: str,
    csv_path: Path,
    project_root: Optional[Path] = None,
    output_subdir: Optional[str] = None,
    **kwargs,
) -> FigureResult:
    """Dispatch graph creation based on dataset and graph name."""
    dataset = dataset.lower().strip()
    graph = graph.lower().strip()
    out = graphs_dir(project_root) / (output_subdir or dataset)
    out.mkdir(parents=True, exist_ok=True)

    if graph in {"cell_count", "detected_cells", "time_population"}:
        return time_population_plot_from_csv(
            csv_path=csv_path,
            output_dir=out,
            value_col=kwargs.get("value_col"),
            time_col=kwargs.get("time_col"),
            group_col=kwargs.get("group_col"),
            stem=kwargs.get("stem", f"{dataset}_{graph}"),
            ylabel=kwargs.get("ylabel", "Detected cells"),
        )

    if graph in {"cell_length", "length"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col", "cell_length_um"), group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_cell_length"), ylabel=kwargs.get("ylabel", "Cell length (µm)"))

    if graph in {"cell_area", "area"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col", "cell_area_um2"), group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_cell_area"), ylabel=kwargs.get("ylabel", "Cell area (µm²)"))

    if graph in {"cell_volume", "volume"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col", "cell_volume_um3"), group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_cell_volume"), ylabel=kwargs.get("ylabel", "Cell volume (µm³)"))

    if graph in {"homogeneity", "wga_homogeneity", "dapi_homogeneity"}:
        return boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col"), group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_{graph}"), ylabel=kwargs.get("ylabel", graph.replace("_", " ").title()))

    if graph in {"axial_profile", "wga_axial", "dapi_axial", "hada_axial"}:
        return axial_profile_from_csv(csv_path, out, value_cols=kwargs.get("value_cols"), axis_col=kwargs.get("axis_col"), group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_{graph}"), ylabel=kwargs.get("ylabel", "Normalized intensity"))

    if graph in {"radial_profile", "radial"}:
        return radial_profile_from_csv(csv_path, out, value_cols=kwargs.get("value_cols"), radius_col=kwargs.get("radius_col"), group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_radial_profile"), ylabel=kwargs.get("ylabel", "Normalized intensity"))

    if graph in {"kymograph", "standardized_map", "standardized_cell_map"}:
        return standardized_cell_map_from_csv(csv_path, out, value_col=kwargs.get("value_col"), axis_col=kwargs.get("axis_col"), group_col=kwargs.get("group_col"), cell_col=kwargs.get("cell_col"), stem=kwargs.get("stem", f"{dataset}_{graph}"), intensity_label=kwargs.get("intensity_label", "Normalized intensity"))

    if graph in {"colocalization", "spatial_relationship", "spatial_relation"}:
        return colocalization_boxplot_from_csv(csv_path, out, value_col=kwargs.get("value_col"), group_col=kwargs.get("group_col"), pair_col=kwargs.get("pair_col"), stem=kwargs.get("stem", f"{dataset}_{graph}"))

    if graph in {"correlation", "scatter"}:
        if not kwargs.get("x_col") or not kwargs.get("y_col"):
            raise ValueError("Correlation plot requires x_col and y_col.")
        return correlation_plot_from_csv(csv_path, out, x_col=kwargs["x_col"], y_col=kwargs["y_col"], group_col=kwargs.get("group_col"), stem=kwargs.get("stem", f"{dataset}_correlation"))

    raise ValueError(f"Unknown graph type: {graph}")
