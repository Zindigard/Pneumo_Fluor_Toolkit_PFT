"""
Verify intensity preservation in saved local-threshold OME-Zarr outputs.

The check compares each saved output against its source OME-Zarr image and the
expected processing result. Normally that result is a freshly recomputed local
threshold. Explicitly excluded THY 120-minute samples are expected to be exact
unchanged pass-through copies. A plane passes only when:

1. source, expected, and saved shapes are identical;
2. the saved dtype equals the source dtype;
3. every retained pixel equals the original value exactly;
4. every rejected pixel equals zero;
5. the complete saved plane equals the freshly recomputed expected plane.

These conditions directly exclude percentile normalization, min-max scaling,
dtype conversion, smoothing, or any other intensity transformation.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/check_local_threshold_intensity.py --help

Representative execution:

    python scripts/denoising/check_local_threshold_intensity.py \
        --dataset 2d_time \
        --level 0 \
        --max-images 4 \
        --image-name WT_HADA_NHS_40min_ROI1_SIM \
        --high-percentile 99.2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

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
    raise RuntimeError("Cannot find the PFT repository root containing scripts and src/PFT")


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.denoising.denoiser import (
    _iter_plane_slices,
    _selected_channels,
    apply_local_threshold_to_array,
    is_passthrough_image,
    list_source_images,
    local_threshold_output_root,
)
from PFT.core_prog_parts.denoising.local_threshold_filter import (
    LocalThresholdParams,
    apply_local_threshold_2d,
    validate_intensity_preservation,
)
from PFT.core_prog_parts.denoising.notch_filter import _ensure_cyx, _to_numpy

DATASETS = ("2d_time", "2d_wga_dapi")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write validation rows as UTF-8 CSV.

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

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_processing_attrs(zarr_path: Path) -> dict[str, Any]:
    """Read the saved processing provenance when zarr is available.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _read_processing_attrs(zarr_path=Path("path/to/resource"))
    """
    try:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        processing = root.attrs.get("pft_processing", {})
        return dict(processing) if isinstance(processing, dict) else {}
    except Exception:
        return {}


def _validate_image(
    dataset: str,
    source_path: Path,
    filtered_path: Path,
    params: LocalThresholdParams,
    *,
    level: int,
) -> list[dict[str, Any]]:
    """Validate every selected plane in one saved image.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_path (Path): Filesystem path associated with source.
        filtered_path (Path): Filesystem path associated with filtered.
        params (LocalThresholdParams): Value specifying params for the operation.
        level (int): Numerical value controlling level.

    Returns:
        list[dict[str, Any]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _validate_image(
        ...     dataset="2d_time",
        ...     source_path=Path("path/to/resource"),
        ...     filtered_path=Path("path/to/resource"),
        ...     params=...,
        ...     level=1,
        ... )
    """
    source_array, source_axes = load_ome_zarr(source_path, level=level, as_numpy=False)
    saved_array, saved_axes = load_ome_zarr(filtered_path, level=0, as_numpy=False)
    source, source_axes = _ensure_cyx(_to_numpy(source_array), source_axes)
    saved, saved_axes = _ensure_cyx(_to_numpy(saved_array), saved_axes)

    passthrough = is_passthrough_image(dataset, source_path.parent.name)
    expected, _ = apply_local_threshold_to_array(
        source,
        source_axes,
        dataset,
        params,
        passthrough=passthrough,
    )
    array_shape_match = source.shape == expected.shape == saved.shape
    axes_match = source_axes == saved_axes
    dtype_match = source.dtype == saved.dtype
    complete_array_match = bool(
        array_shape_match and dtype_match and np.array_equal(expected, saved)
    )

    processing = _read_processing_attrs(filtered_path)
    normalization_attr = processing.get("normalization_applied", "missing")
    normalization_attr_ok = normalization_attr is False
    operation_attr = processing.get("operation", "missing")
    expected_operation = "passthrough_unfiltered" if passthrough else "local_threshold"
    operation_attr_ok = operation_attr == expected_operation
    channels = _selected_channels(dataset, source_axes, source)
    rows: list[dict[str, Any]] = []

    for channel_index, frame_index, plane_slice in _iter_plane_slices(
        source.shape,
        source_axes,
        channels,
    ):
        raw_plane = source[plane_slice]
        saved_plane = saved[plane_slice] if array_shape_match else np.empty((0, 0))
        if passthrough:
            keep_mask = np.ones(raw_plane.shape, dtype=bool)
            retained_fraction = 1.0
            threshold_raw = float("nan")
        else:
            _, keep_mask, threshold_info = apply_local_threshold_2d(raw_plane, params)
            retained_fraction = float(threshold_info["frac_final"])
            threshold_raw = float(threshold_info["high_threshold_raw"])

        integrity = validate_intensity_preservation(raw_plane, saved_plane, keep_mask)
        plane_expected = expected[plane_slice] if array_shape_match else np.empty((0, 0))
        plane_exact_match = bool(
            array_shape_match
            and dtype_match
            and np.array_equal(plane_expected, saved_plane)
        )
        passed = bool(
            axes_match
            and array_shape_match
            and dtype_match
            and complete_array_match
            and normalization_attr_ok
            and operation_attr_ok
            and integrity["status"] == "PASS"
            and plane_exact_match
        )
        rows.append(
            {
                "dataset": dataset,
                "image": source_path.parent.name,
                "processing_mode": expected_operation,
                "filter_applied": not passthrough,
                "source_omezarr": str(source_path),
                "filtered_omezarr": str(filtered_path),
                "channel_index": channel_index,
                "frame_index": frame_index,
                "status": "PASS" if passed else "FAIL",
                "source_axes": source_axes,
                "saved_axes": saved_axes,
                "axes_match": axes_match,
                "source_shape": "x".join(str(value) for value in source.shape),
                "saved_shape": "x".join(str(value) for value in saved.shape),
                "shape_match": array_shape_match,
                "source_dtype": str(source.dtype),
                "saved_dtype": str(saved.dtype),
                "dtype_match": dtype_match,
                "normalization_attribute": normalization_attr,
                "normalization_attribute_ok": normalization_attr_ok,
                "operation_attribute": operation_attr,
                "expected_operation": expected_operation,
                "operation_attribute_ok": operation_attr_ok,
                "complete_array_exact_match": complete_array_match,
                "plane_exact_match": plane_exact_match,
                "kept_mismatch_count": integrity["kept_mismatch_count"],
                "rejected_nonzero_count": integrity["rejected_nonzero_count"],
                "max_abs_kept_error": integrity["max_abs_kept_error"],
                "retained_pixel_fraction": retained_fraction,
                "threshold_raw_intensity": threshold_raw,
            }
        )
    return rows

def _summary_rows(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one compact dataset-level validation summary.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        rows (list[dict[str, Any]]): Text value specifying rows.

    Returns:
        list[dict[str, Any]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _summary_rows(dataset="2d_time", rows="rows")
    """
    failures = [row for row in rows if row["status"] == "FAIL"]
    return [
        {
            "dataset": dataset,
            "n_planes_checked": len(rows),
            "n_planes_passed": len(rows) - len(failures),
            "n_planes_failed": len(failures),
            "status": "PASS" if not failures and rows else "FAIL",
            "maximum_kept_intensity_error": max(
                (float(row["max_abs_kept_error"]) for row in rows),
                default=float("nan"),
            ),
            "total_kept_mismatches": sum(
                max(0, int(row["kept_mismatch_count"])) for row in rows
            ),
            "total_rejected_nonzero_pixels": sum(
                max(0, int(row["rejected_nonzero_count"])) for row in rows
            ),
        }
    ]


def _parse_args() -> argparse.Namespace:
    """Parse saved-output validation options.

    Returns:
        argparse.Namespace: Result produced by the operation.

    Example:
        >>> result = _parse_args()
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check that saved local-threshold OME-Zarr images retain original "
            "intensities at kept pixels and contain zero elsewhere."
        )
    )
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--image-name",
        action="append",
        default=None,
        help=(
            "Validate one exact sample stem. Repeat the option to validate "
            "multiple named samples."
        ),
    )
    parser.add_argument("--high-percentile", type=float, default=99.2)
    parser.add_argument("--frac3", type=float, default=0.40)
    parser.add_argument("--support3", type=float, default=0.70)
    parser.add_argument("--kernel3", type=int, default=3)
    parser.add_argument("--kernel10", type=int, default=10)
    parser.add_argument("--frac10", type=float, default=0.40)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "filter_checks" / "local_threshold_intensity",
    )
    return parser.parse_args()


def main() -> None:
    """Validate all requested saved local-threshold images and write CSV reports.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    args = _parse_args()
    params = LocalThresholdParams(
        high_percentile=args.high_percentile,
        frac3_keep=args.frac3,
        support3_frac=args.support3,
        kernel3=args.kernel3,
        kernel10=args.kernel10,
        frac10_keep=args.frac10,
    )
    params.validate()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    all_summaries: list[dict[str, Any]] = []
    failed_datasets: list[str] = []

    for dataset in datasets:
        source_paths = list_source_images(dataset)
        if args.image_name:
            requested = set(args.image_name)
            source_paths = [
                path for path in source_paths if path.parent.name in requested
            ]
            found = {path.parent.name for path in source_paths}
            missing = sorted(requested - found)
            if missing:
                raise FileNotFoundError(
                    "Requested source image(s) not found: " + ", ".join(missing)
                )
        if args.max_images is not None:
            source_paths = source_paths[: args.max_images]
        dataset_rows: list[dict[str, Any]] = []
        output_base = local_threshold_output_root(dataset, params)

        for source_path in source_paths:
            filtered_path = output_base / source_path.parent.name / "image.ome.zarr"
            if not filtered_path.exists():
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "image": source_path.parent.name,
                        "source_omezarr": str(source_path),
                        "filtered_omezarr": str(filtered_path),
                        "status": "FAIL",
                        "error": "Filtered OME-Zarr output not found",
                        "kept_mismatch_count": -1,
                        "rejected_nonzero_count": -1,
                        "max_abs_kept_error": float("nan"),
                    }
                )
                continue
            try:
                dataset_rows.extend(
                    _validate_image(
                        dataset,
                        source_path,
                        filtered_path,
                        params,
                        level=args.level,
                    )
                )
            except Exception as exc:
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "image": source_path.parent.name,
                        "source_omezarr": str(source_path),
                        "filtered_omezarr": str(filtered_path),
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                        "kept_mismatch_count": -1,
                        "rejected_nonzero_count": -1,
                        "max_abs_kept_error": float("nan"),
                    }
                )

        detailed_csv = args.output_root / dataset / f"intensity_check_{dataset}.csv"
        summary_csv = args.output_root / dataset / f"intensity_check_summary_{dataset}.csv"
        summaries = _summary_rows(dataset, dataset_rows)
        _write_csv(detailed_csv, dataset_rows)
        _write_csv(summary_csv, summaries)
        all_summaries.extend(summaries)
        print(f"Detailed CSV: {detailed_csv}")
        print(f"Summary CSV : {summary_csv}")
        if summaries[0]["status"] != "PASS":
            failed_datasets.append(dataset)

    global_summary = args.output_root / "intensity_check_all_datasets.csv"
    _write_csv(global_summary, all_summaries)
    print(f"Global summary: {global_summary}")
    if failed_datasets:
        raise SystemExit(
            "Intensity preservation failed for: " + ", ".join(failed_datasets)
        )


if __name__ == "__main__":
    main()
