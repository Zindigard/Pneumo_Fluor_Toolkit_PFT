from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as exc:  
    sm = None
    smf = None
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None

try:
    from .results_graphs import (
        COLUMN_ALIASES,
        find_column,
        find_project_root,
        final_values_dir,
        list_csv_files,
        load_csv,
        normalize_column_name,
        normalize_columns,
    )
except ImportError:  # pragma: no cover
    from PFT.core_prog_parts.results_graphs import (
        COLUMN_ALIASES,
        find_column,
        find_project_root,
        final_values_dir,
        list_csv_files,
        load_csv,
        normalize_column_name,
        normalize_columns,
    )


STAT_COLUMN_ALIASES: dict[str, list[str]] = {
    **COLUMN_ALIASES,
    "medium": ["medium", "media", "growth_medium", "culture_medium", "condition"],
    "csp": ["csp", "competence", "csp_treatment", "competence_induction", "induction"],
    "image_id": ["image_id", "image", "file", "filename", "sample", "sample_id", "roi", "roi_id", "stack", "stack_id"],
    "model": ["model", "segmentation_model", "method", "algorithm"],
    "iou": ["iou", "intersection_over_union", "image_iou", "mean_iou"],
    "dice": ["dice", "dice_score", "dice_coefficient", "f1", "f1_score"],
    "precision": ["precision"],
    "recall": ["recall", "sensitivity"],
    "auc": ["auc", "area_under_curve", "area_under_the_curve"],
}


@dataclass
class StatisticsResult:
    """Paths created by one statistical run."""

    primary_csv_path: Path
    prepared_csv_path: Optional[Path] = None
    extra_paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def all_paths(self) -> list[Path]:
        paths = [self.primary_csv_path]
        if self.prepared_csv_path is not None:
            paths.append(self.prepared_csv_path)
        paths.extend(self.extra_paths)
        return paths


def statistics_dir(project_root: Optional[Path] = None) -> Path:
    """Return the output directory for statistical result tables."""
    root = find_project_root(project_root)
    path = root / "results" / "statistics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_column(
    df: pd.DataFrame,
    logical_name: str,
    required: bool = True,
    extra_candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Find a column using the graph aliases plus statistics-specific aliases."""
    cols = set(df.columns)
    candidates: list[str] = []
    if extra_candidates:
        candidates.extend(extra_candidates)
    candidates.extend(STAT_COLUMN_ALIASES.get(logical_name, []))
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


def _save_csv(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def _is_numeric_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() > 0.5


def _discover_value_columns(df: pd.DataFrame, dataset: str, graph: str) -> list[str]:
    """Detect numeric value columns for profile-like wide CSV tables."""
    dataset = dataset.lower().strip()
    graph = graph.lower().strip()

    preferred_keys: list[str] = []
    if "3d" in dataset or "sim" in dataset:
        preferred_keys = ["hada", "nada", "tada"]
    elif "wga" in dataset or "dapi" in dataset:
        preferred_keys = ["wga", "dapi"]
    elif "hada" in dataset or "time" in dataset:
        preferred_keys = ["hada"]

    discovered: list[str] = []
    for key in preferred_keys:
        col = _find_column(df, key, required=False)
        if col and col not in discovered and _is_numeric_series(df[col]):
            discovered.append(col)

    for key in ["value", "homogeneity", "cell_length", "cell_area", "cell_volume", "spatial_relation", "iou", "dice"]:
        col = _find_column(df, key, required=False)
        if col and col not in discovered and _is_numeric_series(df[col]):
            discovered.append(col)

    if not discovered:
        ignore = {"condition", "time", "cell_id", "axis", "radius", "channel", "medium", "csp"}
        for col in df.columns:
            if col in ignore:
                continue
            if _is_numeric_series(df[col]):
                discovered.append(col)

    return discovered


def _classify_3d_source_channel(raw: object) -> str:
    name = normalize_column_name(str(raw))
    if any(key in name for key in ["hada", "blue", "b_channel", "channel_b"]):
        return "old_blue"
    if any(key in name for key in ["nada", "green", "g_channel", "channel_g"]):
        return "old_green"
    if any(key in name for key in ["tada", "red", "r_channel", "channel_r"]):
        return "red"
    return name


def map_3d_channel(raw: object, switch_blue_green: bool = True) -> str:
    """Map HADA/NADA/TADA or blue/green/red labels to final thesis channel labels."""
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


def _map_general_marker(raw: object, dataset: str, switch_blue_green: bool = True) -> str:
    name = normalize_column_name(str(raw))
    if "3d" in dataset.lower() or "sim" in dataset.lower():
        return map_3d_channel(name, switch_blue_green=switch_blue_green)
    if "wga" in name:
        return "wga"
    if "dapi" in name:
        return "dapi"
    if "hada" in name:
        return "hada"
    if "nada" in name:
        return "nada"
    if "tada" in name:
        return "tada"
    return name


def _parse_medium_from_condition(value: object) -> str:
    text = str(value).upper()
    if "THY" in text:
        return "THY"
    if "NHS" in text:
        return "NHS"
    return str(value)


def _parse_csp_from_condition(value: object) -> str:
    text = str(value).upper().replace(" ", "")
    if "+CSP" in text or "CSP+" in text or text.endswith("CSP") and "NO" not in text and "WITHOUT" not in text:
        return "+CSP"
    if "CSP" in text and ("NO" in text or "WITHOUT" in text or "-CSP" in text):
        return "no CSP"
    if "CSP" in text:
        return "+CSP"
    return "no CSP"


def add_derived_factors(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add medium/CSP/channel factors when they can be inferred from existing columns."""
    out = df.copy()

    condition_col = _find_column(out, "condition", required=False)
    channel_col = _find_column(out, "channel", required=False)

    if condition_col is not None:
        if "medium" not in out.columns:
            out["medium"] = out[condition_col].map(_parse_medium_from_condition)
        if "csp" not in out.columns and ("wga" in dataset.lower() or "dapi" in dataset.lower()):
            out["csp"] = out[condition_col].map(_parse_csp_from_condition)

    if channel_col is not None and "3d" in dataset.lower():
        out["channel"] = out[channel_col].map(lambda v: map_3d_channel(v, switch_blue_green=True))

    return out


def _find_existing_factor(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    return _find_column(df, logical_name, required=False)


def infer_default_value_col(df: pd.DataFrame, dataset: str, graph: str, value_col: Optional[str] = None) -> str:
    """Infer the dependent variable for non-profile statistical tests."""
    if value_col:
        col = normalize_column_name(value_col)
        if col not in df.columns:
            raise KeyError(f"Requested value column '{col}' is not present. Available columns: {', '.join(df.columns)}")
        return col

    graph = graph.lower().strip()
    if graph in {"cell_length", "length"}:
        return _find_column(df, "cell_length")
    if graph in {"cell_area", "area"}:
        return _find_column(df, "cell_area")
    if graph in {"cell_volume", "volume"}:
        return _find_column(df, "cell_volume")
    if "homogeneity" in graph:
        return _find_column(df, "homogeneity")
    if graph in {"colocalization", "colocalisation", "spatial_relation", "spatial_relationship"}:
        return _find_column(df, "spatial_relation")
    if graph in {"iou", "dice", "precision", "recall"}:
        return _find_column(df, graph)
    if graph in {"cell_count", "detected_cells", "time_population"}:
        count_col = _find_column(df, "cell_count", required=False)
        if count_col:
            return count_col
    value = _find_column(df, "value", required=False)
    if value:
        return value

    discovered = _discover_value_columns(df, dataset, graph)
    if not discovered:
        raise ValueError("Could not infer a numeric value column. Please provide --value-col.")
    return discovered[0]


def infer_default_factors(
    df: pd.DataFrame,
    dataset: str,
    graph: str,
    factor_a: Optional[str] = None,
    factor_b: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Infer two ANOVA factors from dataset and graph context."""
    if factor_a:
        factor_a = normalize_column_name(factor_a)
    if factor_b:
        factor_b = normalize_column_name(factor_b)
    if factor_a and factor_b:
        return factor_a, factor_b

    dataset_l = dataset.lower().strip()
    graph_l = graph.lower().strip()

    if "time" in dataset_l or "hada" in dataset_l:
        a = _find_existing_factor(df, "condition")
        b = _find_existing_factor(df, "time")
    elif "wga" in dataset_l or "dapi" in dataset_l:
        a = _find_existing_factor(df, "medium") or _find_existing_factor(df, "condition")
        b = _find_existing_factor(df, "csp")
    elif "3d" in dataset_l or "sim" in dataset_l:
        a = _find_existing_factor(df, "condition")
        b = _find_existing_factor(df, "channel")
    else:
        a = _find_existing_factor(df, "condition")
        b = _find_existing_factor(df, "time") or _find_existing_factor(df, "channel") or _find_existing_factor(df, "csp")

    return factor_a or a, factor_b or b


def _profile_graph(graph: str) -> bool:
    graph_l = graph.lower().strip()
    return graph_l in {
        "axial_profile", "hada_axial", "wga_axial", "dapi_axial", "3d_axial", "axial",
        "radial_profile", "radial", "3d_radial", "kymograph", "standardized_map", "standardized_cell_map",
    }


def _coordinate_column_for_profile(
    df: pd.DataFrame,
    graph: str,
    axis_col: Optional[str] = None,
    radius_col: Optional[str] = None,
) -> str:
    graph_l = graph.lower().strip()
    if graph_l in {"radial_profile", "radial", "3d_radial"}:
        return normalize_column_name(radius_col) if radius_col else _find_column(df, "radius")
    return normalize_column_name(axis_col) if axis_col else _find_column(df, "axis")


def _collect_unit_columns(df: pd.DataFrame, cell_col: Optional[str] = None) -> list[str]:
    unit_cols: list[str] = []
    if cell_col:
        col = normalize_column_name(cell_col)
        if col in df.columns:
            unit_cols.append(col)
    else:
        col = _find_column(df, "cell_id", required=False)
        if col:
            unit_cols.append(col)

    for logical in ["image_id"]:
        col = _find_column(df, logical, required=False)
        if col and col not in unit_cols:
            unit_cols.append(col)

    return unit_cols


def _long_profile_table(
    df: pd.DataFrame,
    dataset: str,
    graph: str,
    coordinate_col: str,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    channel_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    switch_blue_green: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Convert profile CSV to long format suitable for AUC calculation."""
    work = df.copy()
    coordinate_col = normalize_column_name(coordinate_col)
    if coordinate_col not in work.columns:
        raise KeyError(f"Coordinate column '{coordinate_col}' is not present in the CSV.")

    condition_col = _find_column(work, "condition", required=False)
    time_col = _find_column(work, "time", required=False)
    medium_col = _find_column(work, "medium", required=False)
    csp_col = _find_column(work, "csp", required=False)
    channel_col = normalize_column_name(channel_col) if channel_col else _find_column(work, "channel", required=False)
    unit_cols = _collect_unit_columns(work, cell_col=cell_col)

    factor_cols = [c for c in [condition_col, time_col, medium_col, csp_col] if c is not None]
    factor_cols = list(dict.fromkeys(factor_cols))

    if value_col:
        value_col = normalize_column_name(value_col)
    else:
        value_col = _find_column(work, "value", required=False)

    id_vars = list(dict.fromkeys([coordinate_col, *unit_cols, *factor_cols]))

    if channel_col is not None and value_col is not None:
        keep = list(dict.fromkeys([*id_vars, channel_col, value_col]))
        long = work[keep].copy().rename(columns={channel_col: "source_channel", value_col: "intensity"})
        long["marker"] = long["source_channel"].map(lambda v: _map_general_marker(v, dataset, switch_blue_green=switch_blue_green))
    else:
        wide_cols = [normalize_column_name(c) for c in value_cols] if value_cols else _discover_value_columns(work, dataset, graph)
        wide_cols = [c for c in wide_cols if c in work.columns and c not in id_vars]
        if not wide_cols:
            raise ValueError("Could not detect profile intensity columns. Provide --value-col or --value-cols.")
        long = work.melt(id_vars=id_vars, value_vars=wide_cols, var_name="source_channel", value_name="intensity")
        long["marker"] = long["source_channel"].map(lambda v: _map_general_marker(v, dataset, switch_blue_green=switch_blue_green))

    long[coordinate_col] = pd.to_numeric(long[coordinate_col], errors="coerce")
    long["intensity"] = pd.to_numeric(long["intensity"], errors="coerce")
    long = long.dropna(subset=[coordinate_col, "intensity"])
    return long, factor_cols, unit_cols


def calculate_auc_table(
    df: pd.DataFrame,
    dataset: str,
    graph: str,
    axis_col: Optional[str] = None,
    radius_col: Optional[str] = None,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    channel_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    switch_blue_green: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate AUC for axial/radial/kymograph profiles.

    Returns a table with one row per cell/image/profile unit and a column `value`,
    which can be used by t-tests and two-way ANOVA.
    """
    coordinate_col = _coordinate_column_for_profile(df, graph, axis_col=axis_col, radius_col=radius_col)
    long, factor_cols, unit_cols = _long_profile_table(
        df=df,
        dataset=dataset,
        graph=graph,
        coordinate_col=coordinate_col,
        value_col=value_col,
        value_cols=value_cols,
        channel_col=channel_col,
        cell_col=cell_col,
        switch_blue_green=switch_blue_green,
    )

    group_cols = list(dict.fromkeys([*unit_cols, *factor_cols, "marker"]))
    if not unit_cols:
        # This still calculates AUC from a mean profile, but only one AUC value per
        # factor combination will be available. Statistical tests require replicate rows.
        group_cols = list(dict.fromkeys([*factor_cols, "marker"]))

    rows: list[dict[str, object]] = []
    for keys, sub in long.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        profile = (
            sub.groupby(coordinate_col, as_index=False)
            .agg(intensity=("intensity", "mean"))
            .sort_values(coordinate_col)
        )
        x = profile[coordinate_col].to_numpy(dtype=float)
        y = profile["intensity"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        try:
            auc = np.trapezoid(y, x=x)
        except AttributeError:  # NumPy < 2.0
            auc = np.trapz(y, x=x)
        row["value"] = float(auc)
        row["measurement"] = f"{graph}_auc"
        row["coordinate"] = coordinate_col
        rows.append(row)

    auc_df = pd.DataFrame(rows)
    if "marker" in auc_df.columns and "channel" not in auc_df.columns:
        auc_df = auc_df.rename(columns={"marker": "channel"})
    return auc_df, unit_cols


def prepare_statistics_table(
    csv_path: Path,
    dataset: str,
    graph: str,
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    axis_col: Optional[str] = None,
    radius_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    pair_col: Optional[str] = None,
    use_auc: Optional[bool] = None,
    switch_blue_green: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a CSV and prepare a common table for t-test/ANOVA.

    The output table always has a numeric column named `value` plus available
    factor columns such as condition, time, medium, csp, channel, model.
    """
    df = load_csv(csv_path)

    # Rename explicit columns to their canonical names when supplied.
    rename_map: dict[str, str] = {}
    for provided, canonical in [
        (group_col, "condition"),
        (time_col, "time"),
        (channel_col, "channel"),
        (pair_col, "channel"),
    ]:
        if provided:
            col = normalize_column_name(provided)
            if col in df.columns:
                rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    df = add_derived_factors(df, dataset)

    if use_auc is None:
        use_auc = _profile_graph(graph)

    if use_auc:
        prepared, unit_cols = calculate_auc_table(
            df=df,
            dataset=dataset,
            graph=graph,
            axis_col=axis_col,
            radius_col=radius_col,
            value_col=value_col,
            value_cols=value_cols,
            channel_col="channel" if "channel" in df.columns else None,
            cell_col=cell_col,
            switch_blue_green=switch_blue_green,
        )
        prepared = add_derived_factors(prepared, dataset)
        return prepared, unit_cols

    val_col = infer_default_value_col(df, dataset=dataset, graph=graph, value_col=value_col)
    keep_cols = [val_col]
    for logical in ["condition", "time", "medium", "csp", "channel", "model", "image_id", "cell_id"]:
        col = _find_column(df, logical, required=False)
        if col and col not in keep_cols:
            keep_cols.append(col)

    prepared = df[keep_cols].copy()
    prepared = prepared.rename(columns={val_col: "value"})
    prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
    prepared = prepared.dropna(subset=["value"])
    prepared = add_derived_factors(prepared, dataset)
    unit_cols = [c for c in ["cell_id", "image_id"] if c in prepared.columns]
    return prepared, unit_cols


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm adjustment for multiple pairwise p-values."""
    p = np.asarray([np.nan if pd.isna(v) else float(v) for v in p_values], dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(p))[0]
    if len(valid_idx) == 0:
        return adjusted.tolist()

    order = valid_idx[np.argsort(p[valid_idx])]
    m = len(order)
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min((m - rank + 1) * p[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * sx + (len(y) - 1) * sy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def welch_t_tests(
    df: pd.DataFrame,
    compare_col: str,
    value_col: str = "value",
    within_cols: Optional[Sequence[str]] = None,
    min_n: int = 2,
) -> pd.DataFrame:
    """Run Welch pairwise t-tests across levels of compare_col.

    """
    compare_col = normalize_column_name(compare_col)
    value_col = normalize_column_name(value_col)
    within_cols = [normalize_column_name(c) for c in (within_cols or []) if c]
    within_cols = [c for c in within_cols if c in df.columns and c != compare_col]

    work = df[[value_col, compare_col, *within_cols]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col, compare_col])
    work[compare_col] = work[compare_col].astype(str)

    if within_cols:
        grouped = work.groupby(within_cols, dropna=False, sort=False)
    else:
        grouped = [((), work)]

    rows: list[dict[str, object]] = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        stratum = {col: value for col, value in zip(within_cols, keys)}
        levels = list(dict.fromkeys(sub[compare_col].astype(str).tolist()))
        if len(levels) < 2:
            continue
        for level_a, level_b in combinations(levels, 2):
            a = pd.to_numeric(sub.loc[sub[compare_col].astype(str) == level_a, value_col], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(sub.loc[sub[compare_col].astype(str) == level_b, value_col], errors="coerce").dropna().to_numpy()
            if len(a) < min_n or len(b) < min_n:
                t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            rows.append({
                **stratum,
                "comparison_variable": compare_col,
                "group_a": level_a,
                "group_b": level_b,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": float(np.mean(a)) if len(a) else np.nan,
                "mean_b": float(np.mean(b)) if len(b) else np.nan,
                "sd_a": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                "sd_b": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                "mean_difference_a_minus_b": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan,
                "cohens_d": _cohens_d(a, b),
                "t_statistic": float(t_stat) if not pd.isna(t_stat) else np.nan,
                "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                "test": "Welch two-sample t-test",
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = _holm_adjust(result["p_value"].tolist())
    return result


def automatic_t_tests(
    df: pd.DataFrame,
    dataset: str,
    graph: str,
    compare_col: Optional[str] = None,
    factor_a: Optional[str] = None,
    factor_b: Optional[str] = None,
    min_n: int = 2,
) -> pd.DataFrame:
    """Run default t-tests for all sensible pairwise comparisons in the table."""
    work = add_derived_factors(df, dataset)

    if compare_col:
        compare_col = normalize_column_name(compare_col)
        within_cols = [c for c in ["condition", "time", "medium", "csp", "channel", "model"] if c in work.columns and c != compare_col]
        return welch_t_tests(work, compare_col=compare_col, within_cols=within_cols, min_n=min_n)

    factor_a, factor_b = infer_default_factors(work, dataset, graph, factor_a=factor_a, factor_b=factor_b)
    results: list[pd.DataFrame] = []
    if factor_a and factor_a in work.columns:
        within = [factor_b] if factor_b and factor_b in work.columns else []
        out = welch_t_tests(work, compare_col=factor_a, within_cols=within, min_n=min_n)
        if not out.empty:
            results.append(out)
    if factor_b and factor_b in work.columns:
        within = [factor_a] if factor_a and factor_a in work.columns else []
        out = welch_t_tests(work, compare_col=factor_b, within_cols=within, min_n=min_n)
        if not out.empty:
            results.append(out)

    if results:
        return pd.concat(results, ignore_index=True)

    fallback = _find_column(work, "condition", required=False) or _find_column(work, "channel", required=False) or _find_column(work, "model", required=False)
    if fallback is None:
        raise ValueError("No suitable grouping column was found for t-tests. Provide --compare-col.")
    return welch_t_tests(work, compare_col=fallback, within_cols=None, min_n=min_n)


def two_way_anova(
    df: pd.DataFrame,
    factor_a: str,
    factor_b: str,
    value_col: str = "value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run two-way ANOVA with interaction and return ANOVA plus group summary."""
    if sm is None or smf is None:  # pragma: no cover
        raise ImportError("statsmodels is required for two-way ANOVA.") from _STATSMODELS_IMPORT_ERROR

    factor_a = normalize_column_name(factor_a)
    factor_b = normalize_column_name(factor_b)
    value_col = normalize_column_name(value_col)

    missing = [c for c in [value_col, factor_a, factor_b] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for two-way ANOVA: {missing}. Available: {list(df.columns)}")

    work = df[[value_col, factor_a, factor_b]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col, factor_a, factor_b])
    work[factor_a] = work[factor_a].astype(str)
    work[factor_b] = work[factor_b].astype(str)

    if work[factor_a].nunique() < 2:
        raise ValueError(f"Two-way ANOVA requires at least two levels in factor A ('{factor_a}').")
    if work[factor_b].nunique() < 2:
        raise ValueError(f"Two-way ANOVA requires at least two levels in factor B ('{factor_b}').")

    model_df = work.rename(columns={value_col: "y", factor_a: "factor_a", factor_b: "factor_b"})
    model = smf.ols("y ~ C(factor_a) * C(factor_b)", data=model_df).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
    anova["term"] = anova["term"].replace({
        "C(factor_a)": factor_a,
        "C(factor_b)": factor_b,
        "C(factor_a):C(factor_b)": f"{factor_a}:{factor_b}",
        "Residual": "residual",
    })
    total_ss = anova["sum_sq"].sum(skipna=True)
    anova["eta_squared"] = anova["sum_sq"] / total_ss if total_ss > 0 else np.nan
    anova["model"] = f"{value_col} ~ {factor_a} + {factor_b} + {factor_a}:{factor_b}"
    anova["test"] = "two-way ANOVA"

    summary = (
        work.groupby([factor_a, factor_b], as_index=False)
        .agg(n=(value_col, "size"), mean=(value_col, "mean"), sd=(value_col, "std"), median=(value_col, "median"))
    )
    return anova, summary


def run_statistics(
    dataset: str,
    graph: str,
    csv_path: Path,
    project_root: Optional[Path] = None,
    output_subdir: Optional[str] = None,
    test: str = "all",
    value_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    axis_col: Optional[str] = None,
    radius_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    channel_col: Optional[str] = None,
    pair_col: Optional[str] = None,
    compare_col: Optional[str] = None,
    factor_a: Optional[str] = None,
    factor_b: Optional[str] = None,
    use_auc: Optional[bool] = None,
    switch_blue_green: bool = True,
    stem: Optional[str] = None,
    min_n: int = 2,
) -> StatisticsResult:
    """Run t-test and/or two-way ANOVA for a selected dataset and graph."""
    dataset = dataset.lower().strip()
    graph = graph.lower().strip()
    test = test.lower().strip().replace("-", "_")
    out = statistics_dir(project_root) / (output_subdir or dataset)
    out.mkdir(parents=True, exist_ok=True)

    prepared, unit_cols = prepare_statistics_table(
        csv_path=csv_path,
        dataset=dataset,
        graph=graph,
        value_col=value_col,
        value_cols=value_cols,
        group_col=group_col,
        time_col=time_col,
        axis_col=axis_col,
        radius_col=radius_col,
        cell_col=cell_col,
        channel_col=channel_col,
        pair_col=pair_col,
        use_auc=use_auc,
        switch_blue_green=switch_blue_green,
    )

    if prepared.empty:
        raise ValueError("Prepared statistics table is empty. Check CSV columns and filters.")

    if stem is None:
        stem = f"{dataset}_{graph}_statistics"

    prepared_path = _save_csv(prepared, out, f"{stem}_prepared_values")
    notes: list[str] = []
    if not unit_cols and (_profile_graph(graph) or use_auc):
        notes.append(
            "AUC was calculated without a cell/image identifier. This usually gives one value per factor combination; "
            "statistical tests need replicate rows. Provide --cell-col or include image_id/roi_id in the CSV."
        )

    paths: list[Path] = []
    primary: Optional[Path] = None

    if test in {"t", "t_test", "ttest", "welch", "all"}:
        t_df = automatic_t_tests(
            prepared,
            dataset=dataset,
            graph=graph,
            compare_col=compare_col,
            factor_a=factor_a,
            factor_b=factor_b,
            min_n=min_n,
        )
        if t_df.empty:
            notes.append("No t-test could be performed because fewer than two comparable groups were available.")
        t_path = _save_csv(t_df, out, f"{stem}_welch_t_tests")
        primary = primary or t_path
        paths.append(t_path)

    if test in {"anova", "two_way_anova", "2way_anova", "two_way", "all"}:
        prepared = add_derived_factors(prepared, dataset)
        fa, fb = infer_default_factors(prepared, dataset=dataset, graph=graph, factor_a=factor_a, factor_b=factor_b)
        if fa is None or fb is None:
            raise ValueError("Could not infer two ANOVA factors. Provide --factor-a and --factor-b.")
        anova_df, summary_df = two_way_anova(prepared, factor_a=fa, factor_b=fb, value_col="value")
        anova_path = _save_csv(anova_df, out, f"{stem}_two_way_anova")
        summary_path = _save_csv(summary_df, out, f"{stem}_two_way_anova_group_summary")
        primary = primary or anova_path
        paths.extend([anova_path, summary_path])

    if primary is None:
        raise ValueError(f"Unknown test option: {test}")

    extra = [p for p in paths if p != primary]
    return StatisticsResult(primary_csv_path=primary, prepared_csv_path=prepared_path, extra_paths=extra, notes=notes)
