"""
Batch local-threshold filtering for the two PFT 2D datasets.

This is the production 2D filtering workflow. It applies only the documented
local-threshold operation. Retained pixels preserve their exact source
intensities and dtype; rejected pixels are set to zero. The quantitative output
is saved as OME-Zarr without normalization. Optional PNG files are explicitly
labeled as display-normalized previews and are not quantitative data. Each
original-versus-filtered comparison uses one shared display scale calculated
from the original plane, which prevents independent contrast stretching from
misrepresenting the filtering effect.

For each processed plane, the workflow also calculates the thesis structured-
noise parameters before and after filtering:

* neighbour correlation;
* Fano factor;
* FFT peak score.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.denoising.fft_diagnostics_2d import (
    channel_label,
    compute_thesis_structured_noise_metrics,
)
from PFT.core_prog_parts.denoising.free_hand_filter import results_filters_dir
from PFT.core_prog_parts.denoising.local_threshold_filter import (
    LocalThresholdParams,
    apply_local_threshold_2d,
    validate_intensity_preservation,
)
from PFT.core_prog_parts.denoising.notch_filter import (
    _ensure_cyx,
    _to_numpy,
    list_omezarr_images,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr

DATASETS = ("2d_time", "2d_wga_dapi")

CURATED_TEST_STEMS: dict[str, list[str]] = {
    "2d_time": [
        "WT_HADA_NHS_5min_ROI1_SIM²",
        "WT_HADA_NHS_5min_ROI4_SIM",
        "WT_HADA_NHS_20min_ROI1_SIM",
        "WT_HADA_NHS_20min_ROI2_SIM",
        "WT_HADA_NHS_20min_ROI3_SIM",
        "WT_HADA_NHS_120min_ROI1_SIM",
        "WT_HADA_NHS_40min_ROI1_SIM²",
        "WT_HADA_NHS_120min_ROI3_SIM²",
        "WT_HADA_THY_20min_ROI2_SIM²",
        "WT_HADA_THY_40min_ROI2_SIM²",
        "WT_HADA_THY_40min_ROI4_SIM²",
        "WT_HADA_THY_120min_ROI1_SIM²",
    ]
}

# These overexposed THY 120-minute images must remain present for the later
# U-Net stage, but local thresholding removes too much useful signal from them.
# They are therefore copied into the filtered-output tree without any numerical
# modification. Both naming variants are included because the dataset contains
# stems with and without the superscript-two suffix.
PASSTHROUGH_IMAGE_STEMS: dict[str, set[str]] = {
    "2d_time": {
        "WT_HADA_THY_120min_ROI1_SIM",
        "WT_HADA_THY_120min_ROI1_SIM²",
        "WT_HADA_THY_120min_ROI2_SIM",
        "WT_HADA_THY_120min_ROI2_SIM²",
    }
}
PASSTHROUGH_REASON = (
    "Overexposed THY 120-minute image retained unchanged for U-Net input continuity"
)


def is_passthrough_image(dataset: str, image_name: str) -> bool:
    """Return whether an image must bypass local thresholding unchanged."""
    return image_name in PASSTHROUGH_IMAGE_STEMS.get(dataset, set())


def list_source_images(dataset: str) -> list[Path]:
    """Return all source OME-Zarr images for a supported 2D dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return list_omezarr_images(dataset)


def list_curated_test_images(dataset: str, zarr_paths: list[Path]) -> list[Path]:
    """Return the curated subset, or all images when no subset is defined."""
    wanted = CURATED_TEST_STEMS.get(dataset)
    if not wanted:
        return zarr_paths
    by_stem = {path.parent.name: path for path in zarr_paths}
    return [by_stem[stem] for stem in wanted if stem in by_stem]


def local_threshold_output_root(dataset: str, params: LocalThresholdParams) -> Path:
    """Return the deterministic output directory for one dataset and parameter set."""
    return (
        results_filters_dir()
        / "Local_high_threshold"
        / dataset
        / "intensity_preserved"
        / params.label
    )


def _selected_channels(dataset: str, axes: str, array: np.ndarray) -> list[int]:
    """Return the biological channels processed for a supported dataset."""
    if "c" not in axes:
        return [0]
    channel_count = array.shape[axes.index("c")]
    if dataset == "2d_time":
        return [0]
    return list(range(min(2, channel_count)))


def _iter_plane_slices(
    shape: tuple[int, ...],
    axes: str,
    channel_indices: list[int],
) -> Iterator[tuple[int, int, tuple[Any, ...]]]:
    """Yield channel, flattened frame index, and a 2D plane slice."""
    axes = axes.lower()
    if not axes.endswith("yx"):
        raise ValueError(f"Expected spatial axes at the end, received axes={axes}")

    channel_axis = axes.index("c") if "c" in axes else None
    frame_axes = [
        index
        for index, axis in enumerate(axes)
        if axis not in {"c", "y", "x"}
    ]
    frame_shape = tuple(shape[index] for index in frame_axes)
    frame_indices = list(np.ndindex(frame_shape)) if frame_shape else [()]

    for channel_index in channel_indices:
        if channel_axis is not None and not 0 <= channel_index < shape[channel_axis]:
            raise IndexError(
                f"Channel {channel_index} outside channel dimension {shape[channel_axis]}"
            )
        for frame_index, coordinates in enumerate(frame_indices):
            plane_slice: list[Any] = [slice(None)] * len(shape)
            if channel_axis is not None:
                plane_slice[channel_axis] = channel_index
            for axis_index, coordinate in zip(frame_axes, coordinates):
                plane_slice[axis_index] = coordinate
            yield channel_index, frame_index, tuple(plane_slice)


def apply_local_threshold_to_array(
    array: np.ndarray,
    axes: str,
    dataset: str,
    params: LocalThresholdParams,
    *,
    passthrough: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Process selected 2D planes by local threshold or unchanged pass-through.

    Parameters
    ----------
    passthrough:
        When ``True``, the returned array is an exact copy of the source array.
        This mode is reserved for explicitly excluded overexposed images that
        must remain available to the later U-Net stage.
    """
    source = np.asarray(array)
    output = source.copy()
    channels = _selected_channels(dataset, axes, source)
    records: list[dict[str, Any]] = []

    for channel_index, frame_index, plane_slice in _iter_plane_slices(
        source.shape,
        axes,
        channels,
    ):
        raw_plane = source[plane_slice]
        if raw_plane.ndim != 2:
            raise ValueError(
                f"Expected a 2D plane after indexing, received shape={raw_plane.shape}"
            )

        if passthrough:
            filtered_plane = np.array(raw_plane, copy=True)
            keep_mask = np.ones(raw_plane.shape, dtype=bool)
            threshold_info: dict[str, float | int | str] = {
                "high_threshold_raw": float("nan"),
                "input_dtype": str(raw_plane.dtype),
                "output_dtype": str(filtered_plane.dtype),
                "n_pixels": int(raw_plane.size),
                "n_high": int(raw_plane.size),
                "n_keep3": int(raw_plane.size),
                "n_strong_cluster": int(raw_plane.size),
                "n_keep10": 0,
                "n_final": int(raw_plane.size),
                "frac_final": 1.0,
            }
            processing_mode = "passthrough_unfiltered"
        else:
            filtered_plane, keep_mask, threshold_info = apply_local_threshold_2d(
                raw_plane,
                params,
            )
            processing_mode = "local_threshold"

        output[plane_slice] = filtered_plane
        integrity = validate_intensity_preservation(
            raw_plane,
            output[plane_slice],
            keep_mask,
        )
        if integrity["status"] != "PASS":
            raise AssertionError(
                "Intensity preservation failed before saving: "
                f"channel={channel_index}, frame={frame_index}, {integrity}"
            )

        raw_metrics = compute_thesis_structured_noise_metrics(raw_plane)
        filtered_metrics = compute_thesis_structured_noise_metrics(filtered_plane)
        row: dict[str, Any] = {
            "channel_index": channel_index,
            "channel": channel_label(dataset, channel_index),
            "frame_index": frame_index,
            "processing_mode": processing_mode,
            "filter_applied": not passthrough,
            "passthrough_reason": PASSTHROUGH_REASON if passthrough else "",
            "input_dtype": str(source.dtype),
            "output_dtype": str(output.dtype),
            **threshold_info,
            **integrity,
        }
        for metric_name, value in raw_metrics.items():
            row[f"raw_{metric_name}"] = value
        for metric_name, value in filtered_metrics.items():
            row[f"filtered_{metric_name}"] = value
            row[f"delta_{metric_name}"] = value - raw_metrics[metric_name]
        records.append(row)

    return output, records

def _source_coordinate_scale(
    zarr_path: Path,
    source_axes: str,
    target_axes: str,
    level: int,
) -> list[float] | None:
    """Read and reorder the source OME-NGFF coordinate scale when available."""
    try:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        multiscales = root.attrs.get("multiscales", [])
        datasets = multiscales[0].get("datasets", []) if multiscales else []
        transformations = datasets[level].get("coordinateTransformations", [])
        scale = next(
            (
                transform.get("scale")
                for transform in transformations
                if transform.get("type") == "scale"
            ),
            None,
        )
        if scale is None or len(scale) != len(source_axes):
            return None
        by_axis = {axis: float(value) for axis, value in zip(source_axes, scale)}
        return [by_axis[axis] for axis in target_axes]
    except Exception:
        return None


def _source_attrs(zarr_path: Path) -> dict[str, Any]:
    """Read selected source attributes for provenance without copying multiscales."""
    try:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        copied: dict[str, Any] = {}
        for key in ("channel_names", "pft_meta", "source_path"):
            if key in root.attrs:
                copied[f"source_{key}"] = root.attrs[key]
        return copied
    except Exception:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV using a stable union of fields."""
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


def _display_limits(
    reference: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
) -> tuple[float, float]:
    """Calculate robust limits for display-only normalization.

    The limits are calculated from the original image plane and then reused for
    both the original and filtered visualizations. Quantitative arrays are not
    modified.
    """
    x = np.asarray(reference, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(finite, [lower_percentile, upper_percentile])
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _display_uint8(
    image: np.ndarray,
    *,
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize one plane to 8-bit for visualization only.

    Parameters
    ----------
    image:
        Quantitative image plane. It is read but never modified.
    limits:
        Optional shared ``(low, high)`` display limits. Passing limits derived
        from the original plane ensures a fair original-versus-filtered
        comparison.
    """
    x = np.asarray(image, dtype=np.float64)
    lo, hi = limits if limits is not None else _display_limits(x)
    scaled = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(255.0 * scaled).astype(np.uint8)


def _save_normalized_comparison_plot(
    path: Path,
    original_display: np.ndarray,
    filtered_display: np.ndarray,
    *,
    channel_name: str,
    threshold_raw: float,
    retained_fraction: float,
    passthrough: bool = False,
) -> None:
    """Save a shared-scale original-versus-output comparison figure."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_title = "Pass-through unchanged" if passthrough else "After local threshold"
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), constrained_layout=True)
    panels = (
        (axes[0], original_display, "Original"),
        (axes[1], filtered_display, output_title),
    )
    for axis, image, title in panels:
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()

    if passthrough:
        subtitle = (
            "excluded from local threshold; saved unchanged; "
            f"retained={100.0 * retained_fraction:.3f}%"
        )
        title = f"{channel_name}: original vs unchanged pass-through"
        footer = (
            "Display-normalized visualization only. Quantitative OME-Zarr is an "
            "exact unchanged copy for this excluded sample."
        )
    else:
        subtitle = (
            f"shared display normalization; raw threshold={threshold_raw:.3f}; "
            f"retained={100.0 * retained_fraction:.3f}%"
        )
        title = f"{channel_name}: original vs locally thresholded"
        footer = (
            "Display-normalized visualization only. Quantitative OME-Zarr "
            "intensities are unchanged at retained pixels."
        )

    figure.suptitle(f"{title}\n{subtitle}", fontsize=12)
    figure.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def _save_first_plane_previews(
    out_dir: Path,
    source: np.ndarray,
    filtered: np.ndarray,
    axes: str,
    dataset: str,
    params: LocalThresholdParams,
    *,
    passthrough: bool = False,
) -> None:
    """Save shared-scale previews, comparison plots, and keep masks."""
    try:
        from PIL import Image
    except ImportError:
        return

    channels = _selected_channels(dataset, axes, source)
    seen: set[int] = set()
    for channel_index, frame_index, plane_slice in _iter_plane_slices(
        source.shape,
        axes,
        channels,
    ):
        if frame_index != 0 or channel_index in seen:
            continue
        seen.add(channel_index)
        raw_plane = source[plane_slice]
        filtered_plane = filtered[plane_slice]
        if passthrough:
            keep_mask = np.ones(raw_plane.shape, dtype=bool)
            threshold_raw = float("nan")
            retained_fraction = 1.0
        else:
            _, keep_mask, threshold_info = apply_local_threshold_2d(raw_plane, params)
            threshold_raw = float(threshold_info["high_threshold_raw"])
            retained_fraction = float(threshold_info["frac_final"])
        label = channel_label(dataset, channel_index)

        shared_limits = _display_limits(raw_plane)
        raw_display = _display_uint8(raw_plane, limits=shared_limits)
        filtered_display = _display_uint8(filtered_plane, limits=shared_limits)

        Image.fromarray(raw_display, mode="L").save(
            out_dir / f"preview_display_normalized_original_{label}.png"
        )
        Image.fromarray(filtered_display, mode="L").save(
            out_dir / f"preview_display_normalized_filtered_{label}.png"
        )
        Image.fromarray((keep_mask.astype(np.uint8) * 255), mode="L").save(
            out_dir / f"mask_keep_{label}.png"
        )
        _save_normalized_comparison_plot(
            out_dir / f"comparison_display_normalized_original_vs_filtered_{label}.png",
            raw_display,
            filtered_display,
            channel_name=label,
            threshold_raw=threshold_raw,
            retained_fraction=retained_fraction,
            passthrough=passthrough,
        )

def process_one_image(
    dataset: str,
    zarr_path: Path,
    params: LocalThresholdParams,
    *,
    level: int = 0,
    save_previews: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    """Process one image by thresholding or explicit unchanged pass-through."""
    array, source_axes = load_ome_zarr(zarr_path, level=level, as_numpy=False)
    source = _to_numpy(array)
    source, axes = _ensure_cyx(source, source_axes)
    passthrough = is_passthrough_image(dataset, zarr_path.parent.name)
    filtered, rows = apply_local_threshold_to_array(
        source,
        axes,
        dataset,
        params,
        passthrough=passthrough,
    )

    if filtered.dtype != source.dtype:
        raise AssertionError(
            f"Output dtype changed from {source.dtype} to {filtered.dtype}"
        )
    if passthrough and not np.array_equal(filtered, source):
        raise AssertionError("Pass-through output differs from the source array")

    out_dir = local_threshold_output_root(dataset, params) / zarr_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    coordinate_scale = _source_coordinate_scale(zarr_path, source_axes, axes, level)
    operation = "passthrough_unfiltered" if passthrough else "local_threshold"
    save_ome_zarr(
        out_dir / "image.ome.zarr",
        filtered,
        axes,
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        coordinate_scale=coordinate_scale,
        extra_attrs={
            **_source_attrs(zarr_path),
            "pft_processing": {
                "operation": operation,
                "filter_applied": not passthrough,
                "passthrough_reason": PASSTHROUGH_REASON if passthrough else "",
                "normalization_applied": False,
                "retained_values_preserved": True,
                "rejected_value": None if passthrough else 0,
                "parameters": params.__dict__,
                "source_omezarr": str(zarr_path),
                "source_level": level,
            },
        },
    )

    image_rows: list[dict[str, Any]] = []
    for row in rows:
        image_rows.append(
            {
                "dataset": dataset,
                "image": zarr_path.parent.name,
                "source_omezarr": str(zarr_path),
                "filtered_omezarr": str(out_dir / "image.ome.zarr"),
                "axes": axes,
                "local_threshold_label": params.label,
                **row,
            }
        )
    _write_csv(out_dir / "local_threshold_metrics.csv", image_rows)
    if save_previews:
        _save_first_plane_previews(
            out_dir,
            source,
            filtered,
            axes,
            dataset,
            params,
            passthrough=passthrough,
        )
    return out_dir, image_rows

def _prompt_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    """Read one validated interactive choice."""
    while True:
        answer = input(prompt).strip() or default
        if answer in choices:
            return choices[answer]
        print(f"Choose one of: {', '.join(choices)}")


def _prompt_float(prompt: str, default: float) -> float:
    """Read one float interactively, using the default for an empty answer."""
    answer = input(prompt).strip()
    return default if not answer else float(answer)


def _parse_args() -> argparse.Namespace:
    """Parse local-threshold batch-processing options."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply intensity-preserving local-threshold filtering to a 2D "
            "dataset and save non-normalized OME-Zarr output."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--mode", choices=("one", "curated", "all"), default=None)
    parser.add_argument("--image-index", type=int, default=None)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--high-percentile", type=float, default=None)
    parser.add_argument("--frac3", type=float, default=None)
    parser.add_argument("--support3", type=float, default=None)
    parser.add_argument("--kernel3", type=int, default=3)
    parser.add_argument("--kernel10", type=int, default=10)
    parser.add_argument("--frac10", type=float, default=None)
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the production local-threshold workflow from the command line."""
    args = _parse_args()
    if args.non_interactive and (args.dataset is None or args.mode is None):
        raise SystemExit("--dataset and --mode are required with --non-interactive")

    dataset = args.dataset
    if dataset is None:
        dataset = _prompt_choice(
            "Dataset [1=2d_time, 2=2d_wga_dapi; default 1]: ",
            {"1": "2d_time", "2": "2d_wga_dapi"},
            "1",
        )
    mode = args.mode
    if mode is None:
        mode = _prompt_choice(
            "Mode [1=one, 2=curated, 3=all; default 1]: ",
            {"1": "one", "2": "curated", "3": "all"},
            "1",
        )

    def resolve(value: float | None, prompt: str, default: float) -> float:
        if value is not None:
            return value
        if args.non_interactive:
            return default
        return _prompt_float(prompt, default)

    params = LocalThresholdParams(
        high_percentile=resolve(
            args.high_percentile,
            "High-intensity percentile [99.2]: ",
            99.2,
        ),
        frac3_keep=resolve(
            args.frac3,
            "3x3 candidate fraction [0.40]: ",
            0.40,
        ),
        support3_frac=resolve(
            args.support3,
            "3x3 retained support fraction [0.70]: ",
            0.70,
        ),
        kernel3=args.kernel3,
        kernel10=args.kernel10,
        frac10_keep=resolve(
            args.frac10,
            "10x10 candidate fraction [0.40]: ",
            0.40,
        ),
    )
    params.validate()

    zarr_paths = list_source_images(dataset)
    if not zarr_paths:
        raise FileNotFoundError(f"No OME-Zarr images found for dataset={dataset}")

    if mode == "all":
        selected = zarr_paths
    elif mode == "curated":
        selected = list_curated_test_images(dataset, zarr_paths)
    else:
        index = args.image_index
        if index is None:
            for position, path in enumerate(zarr_paths):
                print(f"[{position:03d}] {path.parent.name}")
            index = int(input("Image index: ").strip())
        if not 0 <= index < len(zarr_paths):
            raise IndexError(f"image-index must be between 0 and {len(zarr_paths) - 1}")
        selected = [zarr_paths[index]]

    print(f"Filter: {params.human_title}")
    print("Quantitative output normalization: disabled")
    excluded = sorted(PASSTHROUGH_IMAGE_STEMS.get(dataset, set()))
    if excluded:
        print("Unchanged pass-through samples:")
        for stem in excluded:
            print(f"  - {stem}")
    print(f"Processing {len(selected)} image(s) from {dataset}")

    batch_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, zarr_path in enumerate(selected, start=1):
        try:
            out_dir, rows = process_one_image(
                dataset,
                zarr_path,
                params,
                level=args.level,
                save_previews=not args.no_previews,
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset,
                    "image": zarr_path.parent.name,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[FAIL] {zarr_path.parent.name}: {type(exc).__name__}: {exc}")
            continue
        batch_rows.extend(rows)
        mode_label = (
            "PASS-THROUGH UNCHANGED"
            if is_passthrough_image(dataset, zarr_path.parent.name)
            else "LOCAL THRESHOLD"
        )
        print(
            f"[{position:03d}/{len(selected):03d}] {mode_label}: "
            f"{zarr_path.parent.name} -> {out_dir}"
        )

    output_root = local_threshold_output_root(dataset, params)
    _write_csv(output_root / "local_threshold_metrics_all_images.csv", batch_rows)
    _write_csv(output_root / "local_threshold_failures.csv", failures)
    print(f"Batch metrics: {output_root / 'local_threshold_metrics_all_images.csv'}")
    if failures:
        raise SystemExit(f"Local-threshold processing failed for {len(failures)} image(s)")


if __name__ == "__main__":
    main()
