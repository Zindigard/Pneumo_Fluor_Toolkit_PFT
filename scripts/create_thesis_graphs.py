from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Find project root independently of computer-specific path.
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    for parent in [start_path.parent, *start_path.parents]:
        if (parent / "models").exists() and (parent / "scripts").exists():
            return parent

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = find_project_root()

GRAPH_FILES_DIR = PROJECT_ROOT / "results" / "graph_files"
GRAPH_OUTPUT_DIR = PROJECT_ROOT / "results" / "graphs"

COMBINED_PDF_PATH = GRAPH_OUTPUT_DIR / "all_thesis_graphs_combined.pdf"

DARK_BLUE = "#0B2C6B"



def setup_thesis_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


def save_figure(fig, output_dir: Path, filename_stem: str) -> None:
 
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def save_figure_and_collect(
    fig,
    output_dir: Path,
    filename_stem: str,
    figure_collection: list,
) -> None:
    
    save_figure(fig, output_dir, filename_stem)
    figure_collection.append(fig)


def save_combined_pdf(figures: list, output_pdf_path: Path) -> None:
  
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if not figures:
        print("No figures were created. Combined PDF was not saved.")
        return

    with PdfPages(output_pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    print(f"Saved combined PDF: {output_pdf_path}")


def require_csv(csv_path: Path) -> None:
    """
    This script does not create CSV files.
    It only reads already existing CSV files.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nMissing CSV file:\n{csv_path}\n\n"
            "Create CSV files first by running:\n\n"
            "    cd D:\\Thesis\\Pneumo_Fluor_Toolkit_PFT\n"
            "    python models\\_graph_csv_writer.py\n"
        )


def load_csv(csv_filename: str) -> pd.DataFrame:
    csv_path = GRAPH_FILES_DIR / csv_filename
    require_csv(csv_path)
    return pd.read_csv(csv_path)


def scatter_small_dots(
    ax,
    x_position: float,
    values: np.ndarray,
    rng: np.random.Generator,
    jitter_width: float = 0.060,
    dot_size: float = 7,
) -> np.ndarray:
 
    values = np.asarray(values, dtype=float)

    x = np.full(len(values), x_position, dtype=float)
    jitter = rng.uniform(-jitter_width, jitter_width, len(values))
    x_jittered = x + jitter

    ax.scatter(
        x_jittered,
        values,
        s=dot_size,
        color=DARK_BLUE,
        alpha=0.78,
        zorder=3,
    )

    return x_jittered


def scatter_small_dots_with_errorbars(
    ax,
    x_position: float,
    values: np.ndarray,
    errors: np.ndarray,
    rng: np.random.Generator,
    jitter_width: float = 0.060,
    dot_size: float = 7,
) -> np.ndarray:
  
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    x = np.full(len(values), x_position, dtype=float)
    jitter = rng.uniform(-jitter_width, jitter_width, len(values))
    x_jittered = x + jitter

    ax.errorbar(
        x_jittered,
        values,
        yerr=errors,
        fmt="o",
        markersize=np.sqrt(dot_size),
        color=DARK_BLUE,
        ecolor="0.35",
        alpha=0.80,
        elinewidth=0.45,
        capsize=1.5,
        markeredgewidth=0.4,
        zorder=3,
    )

    return x_jittered


def draw_single_boxplot_light(
    ax,
    values: np.ndarray,
    position: float,
    width: float,
    patch_alpha: float = 0.10,
) -> None:
    """
    Draw one boxplot with thin, light whiskers and caps.

    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return

    box = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        showcaps=True,
        whiskerprops={
            "linewidth": 0.55,
            "color": "0.55",
            "alpha": 0.65,
        },
        capprops={
            "linewidth": 0.55,
            "color": "0.55",
            "alpha": 0.65,
        },
        medianprops={
            "linewidth": 1.0,
            "color": "0.30",
        },
        boxprops={
            "linewidth": 0.75,
            "color": "0.45",
            "alpha": 0.65,
        },
    )

    for patch in box["boxes"]:
        patch.set_alpha(patch_alpha)


def draw_two_group_boxplot(
    ax,
    before_values: np.ndarray,
    after_values: np.ndarray,
    before_errors: np.ndarray | None,
    after_errors: np.ndarray | None,
    ylabel: str,
    title: str,
    connect_points: bool = False,
    show_sample_errorbars: bool = False,
    ylim: tuple[float, float] | None = None,
    dot_size: float = 7,
) -> None:
    """
    Two-column boxplot.

    """
    before_values = np.asarray(before_values, dtype=float)
    after_values = np.asarray(after_values, dtype=float)

    before_x = 1.0
    after_x = 2.0

    rng = np.random.default_rng(42)

    draw_single_boxplot_light(
        ax=ax,
        values=before_values,
        position=before_x,
        width=0.22,
        patch_alpha=0.12,
    )

    draw_single_boxplot_light(
        ax=ax,
        values=after_values,
        position=after_x,
        width=0.22,
        patch_alpha=0.12,
    )

    if show_sample_errorbars:
        if before_errors is None or after_errors is None:
            raise ValueError("Sample error bars requested, but error arrays are missing.")

        x_before_j = scatter_small_dots_with_errorbars(
            ax=ax,
            x_position=before_x,
            values=before_values,
            errors=before_errors,
            rng=rng,
            jitter_width=0.070,
            dot_size=dot_size,
        )

        x_after_j = scatter_small_dots_with_errorbars(
            ax=ax,
            x_position=after_x,
            values=after_values,
            errors=after_errors,
            rng=rng,
            jitter_width=0.070,
            dot_size=dot_size,
        )
    else:
        x_before_j = scatter_small_dots(
            ax=ax,
            x_position=before_x,
            values=before_values,
            rng=rng,
            jitter_width=0.070,
            dot_size=dot_size,
        )

        x_after_j = scatter_small_dots(
            ax=ax,
            x_position=after_x,
            values=after_values,
            rng=rng,
            jitter_width=0.070,
            dot_size=dot_size,
        )

    if connect_points:
        for xb, xa, yb, ya in zip(
            x_before_j,
            x_after_j,
            before_values,
            after_values,
        ):
            ax.plot(
                [xb, xa],
                [yb, ya],
                color="0.58",
                linewidth=0.35,
                alpha=0.22,
                zorder=1,
            )

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks([before_x, after_x])
    ax.set_xticklabels(["Before denoising", "After denoising"])
    ax.grid(True, axis="y", linestyle=":", alpha=0.30)

    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_2d_1channel_grouped_before_after_from_csv(
    csv_filename: str,
    title: str,
    filename_stem: str,
    figure_collection: list,
    connect_points: bool = False,
    show_sample_errorbars: bool = False,
) -> None:
    """
    2D timelapse / single-channel SNR graph.

    """
    df = load_csv(csv_filename)

    required_columns = {
        "sample_index",
        "snr_before",
        "snr_after",
        "snr_before_std",
        "snr_after_std",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV file {csv_filename} is missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["sample_index"] = df["sample_index"].astype(int)

    weak_response = {3, 4, 12}
    high_baseline = set(range(17, 23)) | set(range(33, 38))
    intermediate_response = set(range(23, 33))

    high_baseline_mask = df["sample_index"].isin(high_baseline)
    before_other_mask = ~high_baseline_mask

    after_weak_intermediate_mask = (
        df["sample_index"].isin(weak_response)
        | df["sample_index"].isin(intermediate_response)
    )

    after_strong_mask = (
        ~df["sample_index"].isin(weak_response)
        & ~df["sample_index"].isin(high_baseline)
        & ~df["sample_index"].isin(intermediate_response)
    )

    before_x = 1.0
    after_x = 2.0

    before_high_values = df.loc[
        high_baseline_mask,
        "snr_before",
    ].to_numpy(dtype=float)

    before_other_values = df.loc[
        before_other_mask,
        "snr_before",
    ].to_numpy(dtype=float)

    after_weak_intermediate_values = df.loc[
        after_weak_intermediate_mask,
        "snr_after",
    ].to_numpy(dtype=float)

    after_strong_values = df.loc[
        after_strong_mask,
        "snr_after",
    ].to_numpy(dtype=float)

    after_high_values = df.loc[
        high_baseline_mask,
        "snr_after",
    ].to_numpy(dtype=float)

    subgroup_boxplots = [
        {"values": before_high_values, "position": before_x, "box_width": 0.36},
        {"values": before_other_values, "position": before_x, "box_width": 0.28},
        {"values": after_weak_intermediate_values, "position": after_x, "box_width": 0.40},
        {"values": after_strong_values, "position": after_x, "box_width": 0.30},
        {"values": after_high_values, "position": after_x, "box_width": 0.22},
    ]

    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    for group in subgroup_boxplots:
        draw_single_boxplot_light(
            ax=ax,
            values=group["values"],
            position=group["position"],
            width=group["box_width"],
            patch_alpha=0.08,
        )

    rng = np.random.default_rng(42)

    before_x_by_sample = {}
    before_y_by_sample = {}
    after_x_by_sample = {}
    after_y_by_sample = {}

    before_values_all = df["snr_before"].to_numpy(dtype=float)
    after_values_all = df["snr_after"].to_numpy(dtype=float)
    before_errors_all = df["snr_before_std"].to_numpy(dtype=float)
    after_errors_all = df["snr_after_std"].to_numpy(dtype=float)

    x_before = before_x + rng.uniform(-0.13, 0.13, len(df))

    if show_sample_errorbars:
        ax.errorbar(
            x_before,
            before_values_all,
            yerr=before_errors_all,
            fmt="o",
            markersize=np.sqrt(7),
            color=DARK_BLUE,
            ecolor="0.35",
            alpha=0.80,
            elinewidth=0.45,
            capsize=1.5,
            markeredgewidth=0.4,
            zorder=3,
        )
    else:
        ax.scatter(
            x_before,
            before_values_all,
            s=7,
            color=DARK_BLUE,
            alpha=0.78,
            zorder=3,
        )


    x_after = after_x + rng.uniform(-0.14, 0.14, len(df))

    if show_sample_errorbars:
        ax.errorbar(
            x_after,
            after_values_all,
            yerr=after_errors_all,
            fmt="o",
            markersize=np.sqrt(7),
            color=DARK_BLUE,
            ecolor="0.35",
            alpha=0.80,
            elinewidth=0.45,
            capsize=1.5,
            markeredgewidth=0.4,
            zorder=3,
        )
    else:
        ax.scatter(
            x_after,
            after_values_all,
            s=7,
            color=DARK_BLUE,
            alpha=0.78,
            zorder=3,
        )

    for row_i, (_, row) in enumerate(df.iterrows()):
        sample_index = int(row["sample_index"])

        before_x_by_sample[sample_index] = float(x_before[row_i])
        before_y_by_sample[sample_index] = float(row["snr_before"])

        after_x_by_sample[sample_index] = float(x_after[row_i])
        after_y_by_sample[sample_index] = float(row["snr_after"])


    if connect_points:
        for sample_index in before_x_by_sample:
            ax.plot(
                [before_x_by_sample[sample_index], after_x_by_sample[sample_index]],
                [before_y_by_sample[sample_index], after_y_by_sample[sample_index]],
                color="0.58",
                linewidth=0.35,
                alpha=0.22,
                zorder=1,
            )

    ax.set_title(title, pad=10)
    ax.set_ylabel("SNR")

    ax.set_xticks([before_x, after_x])
    ax.set_xticklabels(["Before denoising", "After denoising"])

    ax.set_xlim(0.45, 2.55)
    ax.grid(True, axis="y", linestyle=":", alpha=0.30)

    ax.axvline(
        x=1.50,
        color="0.65",
        linestyle="--",
        linewidth=0.75,
        alpha=0.70,
    )

    fig.tight_layout()
    save_figure_and_collect(fig, GRAPH_OUTPUT_DIR, filename_stem, figure_collection)
    plt.show()


def plot_snr_boxplot_from_csv(
    csv_filename: str,
    title: str,
    filename_stem: str,
    figure_collection: list,
    connect_points: bool,
    show_sample_errorbars: bool = False,
    ylim: tuple[float, float] | None = None,
    dot_size: float = 7,
) -> None:
    """
    Generic SNR boxplot.

    Required CSV columns:

    """
    df = load_csv(csv_filename)

    required_columns = {
        "sample_index",
        "snr_before",
        "snr_after",
        "snr_before_std",
        "snr_after_std",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV file {csv_filename} is missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=(5.0, 4.9))

    draw_two_group_boxplot(
        ax=ax,
        before_values=df["snr_before"].to_numpy(dtype=float),
        after_values=df["snr_after"].to_numpy(dtype=float),
        before_errors=df["snr_before_std"].to_numpy(dtype=float),
        after_errors=df["snr_after_std"].to_numpy(dtype=float),
        ylabel="SNR",
        title=title,
        connect_points=connect_points,
        show_sample_errorbars=show_sample_errorbars,
        ylim=ylim,
        dot_size=dot_size,
    )

    fig.tight_layout()
    save_figure_and_collect(fig, GRAPH_OUTPUT_DIR, filename_stem, figure_collection)
    plt.show()


def plot_iou_boxplot_from_csv(
    csv_filename: str,
    title: str,
    filename_stem: str,
    figure_collection: list,
    show_sample_errorbars: bool = False,
    ylim: tuple[float, float] = (0.48, 0.88),
) -> None:
    """
    IoU distribution graph.

    """
    df = load_csv(csv_filename)

    required_columns = {
        "sample_index",
        "iou",
        "iou_std",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV file {csv_filename} is missing columns: {sorted(missing)}"
        )

    values = df["iou"].to_numpy(dtype=float)
    errors = df["iou_std"].to_numpy(dtype=float)

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(4.4, 4.9))

    iou_x = 1.0

    draw_single_boxplot_light(
        ax=ax,
        values=values,
        position=iou_x,
        width=0.22,
        patch_alpha=0.12,
    )

    if show_sample_errorbars:
        scatter_small_dots_with_errorbars(
            ax=ax,
            x_position=iou_x,
            values=values,
            errors=errors,
            rng=rng,
            jitter_width=0.070,
            dot_size=7,
        )
    else:
        scatter_small_dots(
            ax=ax,
            x_position=iou_x,
            values=values,
            rng=rng,
            jitter_width=0.070,
            dot_size=7,
        )

    ax.set_title(title, pad=10)
    ax.set_ylabel("IoU")
    ax.set_xticks([iou_x])
    ax.set_xticklabels(["Segmentation IoU"])
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle=":", alpha=0.30)

    fig.tight_layout()
    save_figure_and_collect(fig, GRAPH_OUTPUT_DIR, filename_stem, figure_collection)
    plt.show()



def plot_cellpose_finetuning_from_csv(
    csv_filename: str,
    title: str,
    filename_stem: str,
    figure_collection: list,
) -> None:
    """
    Cellpose fine-tuning remains an epoch-based line graph.

    Required CSV columns:
    """
    df = load_csv(csv_filename)

    required_columns = {
        "epoch",
        "iou",
        "iou_std",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV file {csv_filename} is missing columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.errorbar(
        df["epoch"].to_numpy(dtype=float),
        df["iou"].to_numpy(dtype=float),
        yerr=df["iou_std"].to_numpy(dtype=float),
        fmt="o-",
        color=DARK_BLUE,
        capsize=3,
        elinewidth=1,
        markeredgewidth=1,
        linewidth=1.7,
        markersize=4,
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("IoU")
    ax.set_xticks(df["epoch"].to_numpy(dtype=float))
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle=":", alpha=0.30)

    fig.tight_layout()
    save_figure_and_collect(fig, GRAPH_OUTPUT_DIR, filename_stem, figure_collection)
    plt.show()


def main() -> None:
    setup_thesis_style()

    created_figures = []

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading CSV files from: {GRAPH_FILES_DIR}")
    print(f"Saving individual graphs to: {GRAPH_OUTPUT_DIR}")
    print(f"Saving combined PDF to: {COMBINED_PDF_PATH}")



    plot_2d_1channel_grouped_before_after_from_csv(
        csv_filename="graph1_2d_1channel_snr_current.csv",
        title="2D timelapse dataset SNR improvement",
        filename_stem="graph1_2d_timelapse_snr_improvement",
        figure_collection=created_figures,
        connect_points=False,
        show_sample_errorbars=False,
    )

    plot_2d_1channel_grouped_before_after_from_csv(
        csv_filename="graph1_2d_1channel_snr_current.csv",
        title="2D timelapse dataset SNR improvement",
        filename_stem="graph1_2d_timelapse_snr_improvement_sample_errorbars",
        figure_collection=created_figures,
        connect_points=False,
        show_sample_errorbars=True,
    )

    plot_2d_1channel_grouped_before_after_from_csv(
        csv_filename="graph1_2d_1channel_snr_current.csv",
        title="2D timelapse dataset SNR improvement",
        filename_stem="graph1_2d_timelapse_snr_improvement_paired",
        figure_collection=created_figures,
        connect_points=True,
        show_sample_errorbars=False,
    )


    plot_snr_boxplot_from_csv(
        csv_filename="graph2_2d_wga_dapi_snr_current.csv",
        title="2D WGA DAPI SNR improvement",
        filename_stem="graph2_2d_wga_dapi_snr_improvement",
        figure_collection=created_figures,
        connect_points=False,
        show_sample_errorbars=False,
        ylim=None,
        dot_size=6,
    )

    plot_snr_boxplot_from_csv(
        csv_filename="graph2_2d_wga_dapi_snr_current.csv",
        title="2D WGA DAPI SNR improvement",
        filename_stem="graph2_2d_wga_dapi_snr_improvement_sample_errorbars",
        figure_collection=created_figures,
        connect_points=False,
        show_sample_errorbars=True,
        ylim=None,
        dot_size=6,
    )

    plot_snr_boxplot_from_csv(
        csv_filename="graph2_2d_wga_dapi_snr_current.csv",
        title="2D WGA DAPI SNR improvement",
        filename_stem="graph2_2d_wga_dapi_snr_improvement_paired",
        figure_collection=created_figures,
        connect_points=True,
        show_sample_errorbars=False,
        ylim=None,
        dot_size=6,
    )

    plot_snr_boxplot_from_csv(
        csv_filename="graph3_3d_snr_current.csv",
        title="3D dataset SNR improvement across volume",
        filename_stem="graph3_3d_snr_improvement_across_volume",
        figure_collection=created_figures,
        connect_points=False,
        show_sample_errorbars=False,
        ylim=(3.6, 8.9),
        dot_size=7,
    )

    plot_snr_boxplot_from_csv(
        csv_filename="graph3_3d_snr_current.csv",
        title="3D dataset SNR improvement across volume",
        filename_stem="graph3_3d_snr_improvement_across_volume_sample_errorbars",
        figure_collection=created_figures,
        connect_points=False,
        show_sample_errorbars=True,
        ylim=(3.6, 8.9),
        dot_size=7,
    )

    plot_snr_boxplot_from_csv(
        csv_filename="graph3_3d_snr_current.csv",
        title="3D dataset SNR improvement across volume",
        filename_stem="graph3_3d_snr_improvement_across_volume_paired",
        figure_collection=created_figures,
        connect_points=True,
        show_sample_errorbars=False,
        ylim=(3.6, 8.9),
        dot_size=7,
    )

    plot_iou_boxplot_from_csv(
        csv_filename="graph2b_2d_wga_dapi_iou_current.csv",
        title="2D WGA DAPI segmentation",
        filename_stem="graph2b_2d_wga_dapi_segmentation",
        figure_collection=created_figures,
        show_sample_errorbars=False,
        ylim=(0.48, 0.88),
    )

    plot_iou_boxplot_from_csv(
        csv_filename="graph2b_2d_wga_dapi_iou_current.csv",
        title="2D WGA DAPI segmentation",
        filename_stem="graph2b_2d_wga_dapi_segmentation_sample_errorbars",
        figure_collection=created_figures,
        show_sample_errorbars=True,
        ylim=(0.48, 0.88),
    )


    plot_cellpose_finetuning_from_csv(
        csv_filename="graph4_cellpose_finetuning_iou_current.csv",
        title="Cellpose Fine-Tuning IoU",
        filename_stem="graph4_cellpose_finetuning_iou",
        figure_collection=created_figures,
    )


    save_combined_pdf(
        figures=created_figures,
        output_pdf_path=COMBINED_PDF_PATH,
    )

    for fig in created_figures:
        plt.close(fig)


if __name__ == "__main__":
    main()