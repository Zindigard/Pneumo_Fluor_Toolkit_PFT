"""
Calculate raw and post-local-threshold structured-noise parameters for 2D data.

The script reads source OME-Zarr images from both supported 2D datasets. Each
plane is measured twice: once as raw data and once after applying the production
local-threshold filter in memory. Separate CSV files are written for each
dataset.

The thesis-specific CSV reports:

* neighbour correlation, mean and maximum;
* Fano factor, mean and maximum;
* FFT peak score, mean and maximum.

The local-threshold output used for the second measurement is not normalized.
Retained pixels preserve their original intensity values, and rejected pixels
are zero.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/FFT.py --help

Representative execution:

    python scripts/denoising/FFT.py \
        --dataset 2d_time \
        --level 0 \
        --output-root results/example_output \
        --max-images 4 \
        --max-frames 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve()


def _find_project_root() -> Path:
    """Locate the repository root independently of the current directory.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _find_project_root()
    """
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected sibling 'scripts' and "
        "'src/PFT' directories."
    )


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.denoising.denoiser import (
    PASSTHROUGH_IMAGE_STEMS,
    PASSTHROUGH_REASON,
    is_passthrough_image,
)
from PFT.core_prog_parts.denoising.fft_diagnostics_2d import (
    FFTDiagnosticConfig,
    channel_label,
    compute_fft_diagnostics,
    fft_log_magnitude_uint8,
    iter_2d_planes,
)
from PFT.core_prog_parts.denoising.local_threshold_filter import (
    LocalThresholdParams,
    apply_local_threshold_2d,
    validate_intensity_preservation,
)
from PFT.core_prog_parts.denoising.notch_filter import _to_numpy, list_omezarr_images

DATASETS = ("2d_time", "2d_wga_dapi")
STAGES = ("raw", "local_threshold")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as UTF-8 CSV using the union of all column names.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (list[dict[str, Any]]): Text value specifying rows.

    Example:
        >>> _write_csv(path=Path("path/to/resource"), rows="rows")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_preview(path: Path, image_u8: np.ndarray) -> None:
    """Save a display-only grayscale FFT preview.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        image_u8 (np.ndarray): Array containing image u8.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> _save_preview(path=Path("path/to/resource"), image_u8=image_array)
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("FFT preview export requires Pillow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u8, mode="L").save(path)


def _numeric(values: Iterable[Any]) -> np.ndarray:
    """Return finite numeric values as a one-dimensional array.

    Args:
        values (Iterable[Any]): Value specifying values for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _numeric(values=...)
    """
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numbers.append(number)
    return np.asarray(numbers, dtype=np.float64)


def _selected_channel(dataset: str, channel_index: int) -> bool:
    """Return whether a channel belongs to the supported biological dataset.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        channel_index (int): Zero-based index selecting channel.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _selected_channel(dataset="2d_time", channel_index=1)
    """
    if dataset == "2d_time":
        return channel_index == 0
    if dataset == "2d_wga_dapi":
        return channel_index in {0, 1}
    return False


def _summarize_rows(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate detailed rows by processing stage and channel.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        rows (list[dict[str, Any]]): Text value specifying rows.

    Returns:
        list[dict[str, Any]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _summarize_rows(dataset="2d_time", rows="rows")
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["processing_stage"]), str(row["channel"]))].append(row)

    metrics = (
        "neighbour_correlation",
        "row_adjacent_correlation",
        "column_adjacent_correlation",
        "fano_factor",
        "fft_peak_score",
        "fft_peak_to_median",
        "fft_peak_to_p95",
        "fft_peak_period_px",
        "fft_directionality_score",
        "low_frequency_power_fraction",
        "mid_frequency_power_fraction",
        "high_frequency_power_fraction",
        "horizontal_axis_power_fraction",
        "vertical_axis_power_fraction",
        "intensity_mean",
        "intensity_std",
        "intensity_p99_2",
    )

    summaries: list[dict[str, Any]] = []
    for (stage, channel), stage_rows in sorted(grouped.items()):
        flags = [bool(row["structured_noise_flag"]) for row in stage_rows]
        summary: dict[str, Any] = {
            "dataset": dataset,
            "processing_stage": stage,
            "channel": channel,
            "n_planes": len(stage_rows),
            "n_images": len({row["image"] for row in stage_rows}),
            "structured_noise_flag_count": int(sum(flags)),
            "structured_noise_flag_fraction": float(np.mean(flags)) if flags else 0.0,
        }
        for metric in metrics:
            values = _numeric(row.get(metric) for row in stage_rows)
            if values.size:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_maximum"] = float(np.max(values))
        summaries.append(summary)
    return summaries


def _thesis_summary_rows(
    dataset: str,
    detailed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create compact rows matching the structured-noise thesis table.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        detailed_rows (list[dict[str, Any]]): Text value specifying detailed rows.

    Returns:
        list[dict[str, Any]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _thesis_summary_rows(dataset="2d_time", detailed_rows="detailed_rows")
    """
    output: list[dict[str, Any]] = []
    channels = sorted({str(row["channel"]) for row in detailed_rows})
    scopes = [("all_channels", None), *((channel, channel) for channel in channels)]

    for scope_name, selected_channel in scopes:
        for stage in STAGES:
            subset = [
                row
                for row in detailed_rows
                if row["processing_stage"] == stage
                and (
                    selected_channel is None
                    or str(row["channel"]) == selected_channel
                )
            ]
            if not subset:
                continue

            neighbour = _numeric(row["neighbour_correlation"] for row in subset)
            fano = _numeric(row["fano_factor"] for row in subset)
            fft_score = _numeric(row["fft_peak_score"] for row in subset)
            output.append(
                {
                    "dataset": dataset,
                    "scope": scope_name,
                    "processing_stage": stage,
                    "n_images": len({row["image"] for row in subset}),
                    "n_planes": len(subset),
                    "neighbour_correlation_mean": float(np.mean(neighbour)),
                    "neighbour_correlation_maximum": float(np.max(neighbour)),
                    "fano_factor_mean": float(np.mean(fano)),
                    "fano_factor_maximum": float(np.max(fano)),
                    "fft_peak_score_mean": float(np.mean(fft_score)),
                    "fft_peak_score_maximum": float(np.max(fft_score)),
                }
            )
    return output


def _stage_rows(
    *,
    dataset: str,
    image_name: str,
    source_path: Path,
    axes: str,
    channel_index: int,
    frame_index: int,
    raw_plane: np.ndarray,
    threshold_params: LocalThresholdParams,
    fft_config: FFTDiagnosticConfig,
    level: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Calculate raw and actual post-filter-pipeline metrics for one plane.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        image_name (str): Text value specifying image name.
        source_path (Path): Filesystem path associated with source.
        axes (str): Axis specification describing the dimensional order of the image data.
        channel_index (int): Zero-based index selecting channel.
        frame_index (int): Zero-based index selecting frame.
        raw_plane (np.ndarray): Array containing raw plane.
        threshold_params (LocalThresholdParams): Value specifying threshold params for the operation.
        fft_config (FFTDiagnosticConfig): Value specifying Fourier-transform result config for the operation.
        level (int): Numerical value controlling level.

    Returns:
        tuple[list[dict[str, Any]], np.ndarray]: Mapping containing the generated or resolved values.

    Raises:
        AssertionError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _stage_rows(
        ...     dataset="2d_time",
        ...     image_name="image_name",
        ...     source_path=Path("path/to/resource"),
        ...     axes="axes",
        ...     channel_index=1,
        ...     frame_index=1,
        ...     raw_plane=image_array,
        ...     threshold_params=...,
        ...     fft_config=...,
        ...     level=1,
        ... )
    """
    passthrough = is_passthrough_image(dataset, image_name)
    if passthrough:
        filtered = np.array(raw_plane, copy=True)
        keep_mask = np.ones(raw_plane.shape, dtype=bool)
        threshold_info = {
            "frac_final": 1.0,
            "high_threshold_raw": float("nan"),
            "n_high": int(raw_plane.size),
            "n_final": int(raw_plane.size),
        }
        pipeline_operation = "passthrough_unfiltered"
    else:
        filtered, keep_mask, threshold_info = apply_local_threshold_2d(
            raw_plane,
            threshold_params,
        )
        pipeline_operation = "local_threshold"

    integrity = validate_intensity_preservation(raw_plane, filtered, keep_mask)
    if integrity["status"] != "PASS":
        raise AssertionError(
            f"In-memory intensity preservation failed for {dataset}/{image_name}, "
            f"channel={channel_index}, frame={frame_index}: {integrity}"
        )

    channel = channel_label(dataset, channel_index)
    common = {
        "dataset": dataset,
        "image": image_name,
        "source_omezarr": str(source_path),
        "ome_zarr_level": level,
        "axes": axes,
        "channel_index": channel_index,
        "channel": channel,
        "frame_index": frame_index,
        "local_threshold_label": threshold_params.label,
        "local_threshold_high_percentile": threshold_params.high_percentile,
        "local_threshold_frac3_keep": threshold_params.frac3_keep,
        "local_threshold_support3_frac": threshold_params.support3_frac,
        "local_threshold_kernel3": threshold_params.kernel3,
        "local_threshold_kernel10": threshold_params.kernel10,
        "local_threshold_frac10_keep": threshold_params.frac10_keep,
    }

    raw_metrics = compute_fft_diagnostics(raw_plane, fft_config)
    filtered_metrics = compute_fft_diagnostics(filtered, fft_config)
    raw_row = {
        **common,
        "processing_stage": "raw",
        "pipeline_operation": "raw",
        "filter_applied": False,
        "passthrough_reason": "",
        "retained_pixel_fraction": 1.0,
        "intensity_preservation_status": "not_applicable",
        **raw_metrics,
    }
    filtered_row = {
        **common,
        "processing_stage": "local_threshold",
        "pipeline_operation": pipeline_operation,
        "filter_applied": not passthrough,
        "passthrough_reason": PASSTHROUGH_REASON if passthrough else "",
        "retained_pixel_fraction": threshold_info["frac_final"],
        "threshold_raw_intensity": threshold_info["high_threshold_raw"],
        "n_high_candidates": threshold_info["n_high"],
        "n_retained_pixels": threshold_info["n_final"],
        "intensity_preservation_status": integrity["status"],
        "kept_mismatch_count": integrity["kept_mismatch_count"],
        "rejected_nonzero_count": integrity["rejected_nonzero_count"],
        "max_abs_kept_error": integrity["max_abs_kept_error"],
        **filtered_metrics,
    }
    return [raw_row, filtered_row], filtered

def _process_dataset(
    dataset: str,
    output_root: Path,
    fft_config: FFTDiagnosticConfig,
    threshold_params: LocalThresholdParams,
    *,
    level: int,
    max_images: int | None,
    max_frames: int | None,
    save_previews: bool,
) -> tuple[Path, Path, Path, int]:
    """Measure one dataset and write detailed, summary, and thesis CSV files.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        output_root (Path): Directory used for output.
        fft_config (FFTDiagnosticConfig): Value specifying Fourier-transform result config for the operation.
        threshold_params (LocalThresholdParams): Value specifying threshold params for the operation.
        level (int): Numerical value controlling level.
        max_images (int | None): Maximum permitted value of images.
        max_frames (int | None): Maximum permitted value of frames.
        save_previews (bool): Boolean flag controlling save previews.

    Returns:
        tuple[Path, Path, Path, int]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _process_dataset(
        ...     dataset="2d_time",
        ...     output_root=Path("path/to/resource"),
        ...     fft_config=...,
        ...     threshold_params=...,
        ...     level=1,
        ...     max_images=1,
        ...     max_frames=1,
        ...     save_previews=True,
        ... )
    """
    zarr_paths = list_omezarr_images(dataset)
    if max_images is not None:
        zarr_paths = zarr_paths[:max_images]
    if not zarr_paths:
        raise FileNotFoundError(
            f"No OME-Zarr images found for {dataset}. Expected "
            f"results/img/{dataset}/<sample>/image.ome.zarr"
        )

    dataset_dir = output_root / dataset
    detailed_rows: list[dict[str, Any]] = []

    for image_number, zarr_path in enumerate(zarr_paths, start=1):
        array, axes = load_ome_zarr(zarr_path, level=level, as_numpy=False)
        array_np = _to_numpy(array)

        for channel_index, frame_index, raw_plane in iter_2d_planes(
            array_np,
            axes,
            max_frames=max_frames,
        ):
            if not _selected_channel(dataset, channel_index):
                continue
            rows, filtered_plane = _stage_rows(
                dataset=dataset,
                image_name=zarr_path.parent.name,
                source_path=zarr_path,
                axes=axes,
                channel_index=channel_index,
                frame_index=frame_index,
                raw_plane=raw_plane,
                threshold_params=threshold_params,
                fft_config=fft_config,
                level=level,
            )
            detailed_rows.extend(rows)

            if save_previews and frame_index == 0:
                channel = channel_label(dataset, channel_index)
                preview_dir = dataset_dir / "previews" / zarr_path.parent.name
                _save_preview(
                    preview_dir / f"fft_raw_{channel}.png",
                    fft_log_magnitude_uint8(raw_plane),
                )
                _save_preview(
                    preview_dir / f"fft_local_threshold_{channel}.png",
                    fft_log_magnitude_uint8(filtered_plane),
                )

        print(
            f"[{dataset}] {image_number:03d}/{len(zarr_paths):03d} "
            f"{zarr_path.parent.name}"
        )

    detailed_csv = dataset_dir / f"fft_metrics_{dataset}.csv"
    summary_csv = dataset_dir / f"fft_summary_{dataset}.csv"
    thesis_csv = dataset_dir / f"thesis_structured_noise_{dataset}.csv"
    _write_csv(detailed_csv, detailed_rows)
    _write_csv(summary_csv, _summarize_rows(dataset, detailed_rows))
    _write_csv(thesis_csv, _thesis_summary_rows(dataset, detailed_rows))

    parameters_path = dataset_dir / f"fft_parameters_{dataset}.json"
    parameters_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "purpose": "raw versus local-threshold structured-noise comparison",
                "thesis_parameter_definitions": {
                    "neighbour_correlation": (
                        "mean horizontal/vertical adjacent-pixel Pearson correlation"
                    ),
                    "fano_factor": "full-plane variance divided by full-plane mean intensity",
                    "fft_peak_score": (
                        "maximum non-DC Fourier power divided by median non-DC Fourier power"
                    ),
                },
                "fft_diagnostic_config": fft_config.__dict__,
                "local_threshold_parameters": threshold_params.__dict__,
                "passthrough_image_stems": {
                    key: sorted(value)
                    for key, value in PASSTHROUGH_IMAGE_STEMS.items()
                },
                "passthrough_reason": PASSTHROUGH_REASON,
                "normalization_applied": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return detailed_csv, summary_csv, thesis_csv, len(detailed_rows)


def _parse_args() -> argparse.Namespace:
    """Parse dataset, FFT, and local-threshold parameters.

    Returns:
        argparse.Namespace: Result produced by the operation.

    Example:
        >>> result = _parse_args()
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calculate thesis structured-noise parameters before and after "
            "intensity-preserving local-threshold filtering."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to analyze. Default: both 2D datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "fft",
        help="Root directory for separate per-dataset CSV files.",
    )
    parser.add_argument("--level", type=int, default=0, help="OME-Zarr pyramid level.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dc-radius", type=int, default=8)
    parser.add_argument("--peak-ratio-threshold", type=float, default=8.0)
    parser.add_argument("--directionality-threshold", type=float, default=0.15)
    parser.add_argument("--high-percentile", type=float, default=99.2)
    parser.add_argument("--frac3", type=float, default=0.40)
    parser.add_argument("--support3", type=float, default=0.70)
    parser.add_argument("--kernel3", type=int, default=3)
    parser.add_argument("--kernel10", type=int, default=10)
    parser.add_argument("--frac10", type=float, default=0.40)
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Do not save display-only FFT PNG previews.",
    )
    return parser.parse_args()


def main() -> None:
    """Run raw and post-threshold diagnostics for the selected datasets.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    args = _parse_args()
    fft_config = FFTDiagnosticConfig(
        dc_radius_px=args.dc_radius,
        peak_ratio_threshold=args.peak_ratio_threshold,
        directionality_threshold=args.directionality_threshold,
    )
    fft_config.validate()
    threshold_params = LocalThresholdParams(
        high_percentile=args.high_percentile,
        frac3_keep=args.frac3,
        support3_frac=args.support3,
        kernel3=args.kernel3,
        kernel10=args.kernel10,
        frac10_keep=args.frac10,
    )
    threshold_params.validate()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for dataset in datasets:
        try:
            detailed_csv, summary_csv, thesis_csv, n_rows = _process_dataset(
                dataset,
                args.output_root,
                fft_config,
                threshold_params,
                level=args.level,
                max_images=args.max_images,
                max_frames=args.max_frames,
                save_previews=not args.no_previews,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"{dataset}: {message}")
            run_rows.append(
                {
                    "dataset": dataset,
                    "status": "FAIL",
                    "n_detailed_rows": 0,
                    "detailed_csv": "",
                    "summary_csv": "",
                    "thesis_csv": "",
                    "error": message,
                }
            )
            print(f"[FAIL] {dataset}: {message}")
            continue

        run_rows.append(
            {
                "dataset": dataset,
                "status": "PASS",
                "n_detailed_rows": n_rows,
                "detailed_csv": str(detailed_csv),
                "summary_csv": str(summary_csv),
                "thesis_csv": str(thesis_csv),
                "error": "",
            }
        )
        print(f"Detailed CSV: {detailed_csv}")
        print(f"Summary CSV : {summary_csv}")
        print(f"Thesis CSV  : {thesis_csv}")

    run_csv = args.output_root / "fft_run_summary.csv"
    _write_csv(run_csv, run_rows)
    print(f"Run summary : {run_csv}")
    if failures:
        raise SystemExit("Structured-noise diagnostics failed for: " + "; ".join(failures))


if __name__ == "__main__":
    main()
