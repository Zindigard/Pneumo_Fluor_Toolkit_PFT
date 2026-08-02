"""
Create thesis-ready plots for the two-dimensional Noise2Void evaluation.

The command combines two result sources:

1. ``models/<model_name>/pft_training/training_history.csv`` for training and
   validation loss curves; and
2. ``results/noise_analysis/2d/n2v_raw_pair_comparison.csv`` produced by
   ``check_2d.py --mode n2v`` for paired raw/N2V image metrics.

Only the 2D time-lapse HADA and 2D WGA-DAPI datasets are included.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/create_n2v_result_plots.py --help

Representative execution:

    python scripts/denoising/create_n2v_result_plots.py \
        --comparison-csv results/noise_analysis/2d/n2v_raw_pair_comparison.csv \
        --output-dir results/example_output \
        --training-only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_FILE = Path(__file__).resolve()


def find_project_root() -> Path:
    """Locate the PFT repository root from the script location.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_project_root()
    """

    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT repository root.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.denoising.n2v_workflow import MODEL_SPECS, utc_now_iso  # noqa: E402


DATASET_LABELS = {
    "2d_time": "2D time-lapse HADA",
    "2d_wga_dapi": "2D WGA-DAPI",
}


def safe_name(value: str) -> str:
    """Convert a dataset/variant label to a filesystem-safe lowercase name.

    Args:
        value (str): Value to validate, transform, store, or forward.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = safe_name(value="value")
    """

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def read_numeric_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file and return rows as dictionaries.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        list[dict[str, str]]: Mapping containing the generated or resolved values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = read_numeric_csv(path=Path("path/to/resource"))
    """

    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def float_value(row: dict[str, str], key: str) -> float:
    """Convert one CSV field to a finite float.

    Args:
        row (dict[str, str]): Text value specifying row.
        key (str): Key used to access or identify an entry in a mapping.

    Returns:
        float: Computed numerical result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = float_value(row="row", key="key")
    """

    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite value in column {key}: {row[key]!r}")
    return value


def save_training_curve(history_csv: Path, output_dir: Path, model_name: str) -> Path:
    """Create a thesis-ready PNG and PDF loss curve from model history.

    The recorded losses are plotted without smoothing. A logarithmic Y axis is
    used when all displayed losses are positive, which preserves early training
    spikes while making late-stage convergence visible.

    Args:
        history_csv (Path): Filesystem path used for history CSV data.
        output_dir (Path): Directory where generated resources are written.
        model_name (str): Human-readable model name used in output paths and reports.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = save_training_curve(
        ...     history_csv=Path("path/to/resource"),
        ...     output_dir=Path("path/to/resource"),
        ...     model_name="model_name",
        ... )
    """

    from matplotlib.ticker import MaxNLocator

    rows = read_numeric_csv(history_csv)
    epochs = np.asarray([int(float(row["epoch"])) for row in rows], dtype=int)
    loss = (
        np.asarray([float_value(row, "loss") for row in rows], dtype=float)
        if "loss" in rows[0]
        else None
    )
    validation = (
        np.asarray([float_value(row, "val_loss") for row in rows], dtype=float)
        if "val_loss" in rows[0]
        and all(row.get("val_loss", "") != "" for row in rows)
        else None
    )

    display_titles = {
        "n2v_2d_time_raw": "Noise2Void training — 2D HADA time-series model",
        "n2v_2d_wga_dapi_dapi_blue_raw": "Noise2Void training — DAPI-channel model",
        "n2v_2d_wga_dapi_wga_green_raw": "Noise2Void training — WGA-channel model",
        "n2v_2d_wga_dapi_joint_dapi+wga_raw": (
            "Noise2Void training — joint DAPI–WGA model"
        ),
    }
    title = display_titles.get(model_name.lower(), model_name)

    figure = plt.figure(figsize=(7.6, 4.8), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)

    if loss is not None:
        axis.plot(
            epochs,
            loss,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label="Training loss",
        )

    if validation is not None:
        axis.plot(
            epochs,
            validation,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label="Validation loss",
        )

        best = int(np.nanargmin(validation))
        best_epoch = int(epochs[best])

        # Dashed vertical line indicates the epoch with the lowest validation loss.
        # No point marker or textual annotation is added to keep the figure clean.
        axis.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )

    plotted_arrays = [values for values in (loss, validation) if values is not None]
    if plotted_arrays and all(np.all(np.isfinite(values) & (values > 0)) for values in plotted_arrays):
        axis.set_yscale("log")
        axis.set_ylabel("Loss (log scale)")
    else:
        axis.set_ylabel("Loss")

    axis.set_xlabel("Epoch")
    axis.set_title(title, pad=10)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlim(float(np.min(epochs)) - 0.5, float(np.max(epochs)) + 1.0)
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    if axis.lines:
        axis.legend(frameon=False, loc="upper right")

    stem = f"training_curve_{safe_name(model_name)}"
    png = output_dir / f"{stem}.png"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)
    return png


def grouped_comparison_rows(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group comparison records by 2D dataset and N2V output variant.

    Args:
        rows (Sequence[dict[str, str]]): Text value specifying rows.

    Returns:
        dict[tuple[str, str], list[dict[str, str]]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = grouped_comparison_rows(rows="rows")
    """

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset not in DATASET_LABELS:
            continue
        variant = row.get("variant", "unknown")
        groups.setdefault((dataset, variant), []).append(row)
    return groups


def save_paired_snr_plot(rows: Sequence[dict[str, str]], output_dir: Path, dataset: str, variant: str) -> Path:
    """Plot each sample's raw and N2V SNR as a connected pair.

    Args:
        rows (Sequence[dict[str, str]]): Text value specifying rows.
        output_dir (Path): Directory where generated resources are written.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        variant (str): Text value specifying variant.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = save_paired_snr_plot(
        ...     rows="rows",
        ...     output_dir=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     variant="variant",
        ... )
    """

    ordered = sorted(rows, key=lambda row: row.get("sample", ""))
    raw = np.asarray([float_value(row, "raw_roi_snr") for row in ordered])
    denoised = np.asarray([float_value(row, "n2v_roi_snr") for row in ordered])

    figure = plt.figure(figsize=(7, 5), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    for raw_value, denoised_value in zip(raw, denoised):
        axis.plot([0, 1], [raw_value, denoised_value], marker="o", alpha=0.65)
    axis.set_xticks([0, 1], ["Raw", "N2V"])
    axis.set_ylabel("ROI signal-to-noise ratio")
    axis.set_title(f"{DATASET_LABELS[dataset]}: {variant}")
    axis.grid(True, axis="y", alpha=0.3)

    stem = f"paired_snr_{safe_name(dataset)}_{safe_name(variant)}"
    png = output_dir / f"{stem}.png"
    figure.savefig(png, dpi=300)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)
    return png


def save_distribution_plot(
    values: np.ndarray,
    *,
    output_dir: Path,
    dataset: str,
    variant: str,
    metric_name: str,
    y_label: str,
    zero_reference: bool,
) -> Path:
    """Create a box-and-point distribution plot for one N2V metric.

    Args:
        values (np.ndarray): Array containing values.
        output_dir (Path): Directory where generated resources are written.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        variant (str): Text value specifying variant.
        metric_name (str): Text value specifying metric name.
        y_label (str): Text value specifying y label.
        zero_reference (bool): Boolean flag controlling zero reference.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = save_distribution_plot(
        ...     values=image_array,
        ...     output_dir=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     variant="variant",
        ...     metric_name="metric_name",
        ...     y_label="y_label",
        ...     zero_reference=True,
        ... )
    """

    figure = plt.figure(figsize=(5.5, 5), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.boxplot(values, positions=[1], widths=0.35, showmeans=True)
    rng = np.random.default_rng(0)
    x = 1.0 + rng.uniform(-0.07, 0.07, size=len(values))
    axis.scatter(x, values, alpha=0.75)
    if zero_reference:
        axis.axhline(0.0, linestyle="--", linewidth=1)
    axis.set_xticks([1], [variant])
    axis.set_ylabel(y_label)
    axis.set_title(DATASET_LABELS[dataset])
    axis.grid(True, axis="y", alpha=0.3)

    stem = f"{safe_name(metric_name)}_{safe_name(dataset)}_{safe_name(variant)}"
    png = output_dir / f"{stem}.png"
    figure.savefig(png, dpi=300)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)
    return png


def paired_wilcoxon(raw: np.ndarray, denoised: np.ndarray) -> tuple[float | None, float | None, str]:
    """Return a paired Wilcoxon statistic and p-value when SciPy is available.

    Args:
        raw (np.ndarray): Array containing raw.
        denoised (np.ndarray): Array containing denoised.

    Returns:
        tuple[float | None, float | None, str]: Collection containing the generated or selected values.

    Example:
        >>> result = paired_wilcoxon(raw=image_array, denoised=image_array)
    """

    if len(raw) < 2:
        return None, None, "not calculated: fewer than two pairs"
    if np.allclose(raw, denoised):
        return 0.0, 1.0, "all paired differences are zero"
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None, None, "not calculated: scipy is unavailable"
    result = wilcoxon(denoised, raw, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue), "two-sided paired Wilcoxon signed-rank test"


def descriptive_record(dataset: str, variant: str, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Calculate descriptive metrics and an optional paired significance test.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        variant (str): Text value specifying variant.
        rows (Sequence[dict[str, str]]): Text value specifying rows.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = descriptive_record(
        ...     dataset="2d_time",
        ...     variant="variant",
        ...     rows="rows",
        ... )
    """

    raw = np.asarray([float_value(row, "raw_roi_snr") for row in rows], dtype=float)
    denoised = np.asarray([float_value(row, "n2v_roi_snr") for row in rows], dtype=float)
    delta = np.asarray([float_value(row, "delta_roi_snr") for row in rows], dtype=float)
    ssim = np.asarray([float_value(row, "ssim_raw_n2v") for row in rows], dtype=float)
    statistic, p_value, test_note = paired_wilcoxon(raw, denoised)
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "variant": variant,
        "n": int(len(rows)),
        "raw_snr_mean": float(np.mean(raw)),
        "raw_snr_standard_deviation": float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0,
        "n2v_snr_mean": float(np.mean(denoised)),
        "n2v_snr_standard_deviation": float(np.std(denoised, ddof=1)) if len(denoised) > 1 else 0.0,
        "delta_snr_mean": float(np.mean(delta)),
        "delta_snr_standard_deviation": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
        "delta_snr_median": float(np.median(delta)),
        "ssim_mean": float(np.mean(ssim)),
        "ssim_standard_deviation": float(np.std(ssim, ddof=1)) if len(ssim) > 1 else 0.0,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "wilcoxon_note": test_note,
    }


def write_summary(records: Sequence[dict[str, Any]], output_dir: Path, figures: Sequence[Path]) -> Path:
    """Write descriptive statistics and a manifest of generated figure files.

    Args:
        records (Sequence[dict[str, Any]]): Text value specifying records.
        output_dir (Path): Directory where generated resources are written.
        figures (Sequence[Path]): Filesystem path used for figures.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = write_summary(
        ...     records="records",
        ...     output_dir=Path("path/to/resource"),
        ...     figures=Path("path/to/resource"),
        ... )
    """

    csv_path = output_dir / "n2v_thesis_plot_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "n2v_thesis_plot_statistics.json").write_text(
        json.dumps(list(records), indent=2), encoding="utf-8"
    )
    lines = [
        "PFT 2D Noise2Void thesis-plot report",
        "=" * 72,
        f"Generated (UTC): {utc_now_iso()}",
        "Datasets: 2D time-lapse HADA and 2D WGA-DAPI only",
        f"Statistic groups: {len(records)}",
        f"Figures generated: {len(figures)}",
        "",
    ]
    for record in records:
        lines.append(
            f"{record['dataset']} | {record['variant']} | n={record['n']} | "
            f"raw SNR={record['raw_snr_mean']:.6g} | N2V SNR={record['n2v_snr_mean']:.6g} | "
            f"delta={record['delta_snr_mean']:.6g} | SSIM={record['ssim_mean']:.6g} | "
            f"p={record['wilcoxon_p_value'] if record['wilcoxon_p_value'] is not None else '-'}"
        )
    lines.extend(["", "Generated figures", "-----------------"])
    lines.extend(str(path) for path in figures)
    report = output_dir / "N2V_THESIS_PLOTS_REPORT.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for input CSV and output directory selection.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """

    parser = argparse.ArgumentParser(
        description="Create 2D N2V training and raw-versus-denoised result plots for the thesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "noise_analysis" / "2d" / "n2v_raw_pair_comparison.csv",
        help="Paired raw/N2V metrics generated by check_2d.py --mode n2v.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "N2V" / "thesis_plots",
        help="Directory for PNG, PDF, CSV, JSON, and TXT outputs.",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help=(
            "Generate only the model training curves and skip all SNR, SSIM, "
            "statistics, and comparison-CSV processing."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Create available training curves and all 2D comparison plots.

    Args:
        argv (list[str] | None): Optional command-line argument sequence. When omitted, arguments are read from ``sys.argv``. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """

    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    for spec in MODEL_SPECS.values():
        history = (
            PROJECT_ROOT
            / "models"
            / spec.model_name
            / "pft_training"
            / "training_history.csv"
        )
        if history.is_file():
            figures.append(save_training_curve(history, output_dir, spec.model_name))
        else:
            print(f"[WARN] Training history not found for {spec.model_name}: {history}")

    if args.training_only:
        print(f"N2V training curves: {output_dir}")
        print(f"Training-curve figures generated: {len(figures)}")
        return 0

    comparison_rows = read_numeric_csv(args.comparison_csv.resolve())
    groups = grouped_comparison_rows(comparison_rows)
    if not groups:
        raise SystemExit("The comparison CSV contains no supported 2D N2V records.")

    statistics: list[dict[str, Any]] = []
    for (dataset, variant), rows in sorted(groups.items()):
        figures.append(save_paired_snr_plot(rows, output_dir, dataset, variant))
        delta = np.asarray([float_value(row, "delta_roi_snr") for row in rows], dtype=float)
        ssim = np.asarray([float_value(row, "ssim_raw_n2v") for row in rows], dtype=float)
        figures.append(
            save_distribution_plot(
                delta,
                output_dir=output_dir,
                dataset=dataset,
                variant=variant,
                metric_name="delta_snr",
                y_label="N2V SNR minus raw SNR",
                zero_reference=True,
            )
        )
        figures.append(
            save_distribution_plot(
                ssim,
                output_dir=output_dir,
                dataset=dataset,
                variant=variant,
                metric_name="ssim",
                y_label="SSIM between raw and N2V",
                zero_reference=False,
            )
        )
        statistics.append(descriptive_record(dataset, variant, rows))

    report = write_summary(statistics, output_dir, figures)
    print(f"N2V thesis plots: {output_dir}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())