"""
Run two-dimensional Noise2Void inference on one or all original images.

Despite the historical filename, this command is the PFT N2V inference entry
point.  It reads only original OME-Zarr images from
``results/img/<dataset>/<sample>/image.ome.zarr`` and writes one raw-range
``float32`` prediction to the standard location::

    results/N2V/<dataset>/<sample>/<variant>/denoised/image.ome.zarr

The command can process one named sample or every image required by the selected
model.  A single source frame is selected because the existing ``check_2d.py``
analysis pairs one N2V output with one original frame and its corresponding
manual mask.

Every output stores:

* the complete original OME-Zarr root attributes and a SHA-256 metadata hash;
* copied CZI/PFT metadata and physical coordinate scales;
* source dataset, sample, frame, channels, axes, shape, dtype, and chunks;
* N2V model key, model folder name, output axes, shape, and dtype;
* explicit declarations that input and output normalization were ``none``; and
* a UTC processing timestamp.

The prediction array itself is never percentile-normalized or min-max-normalized.
Display normalization is used only for optional comparison figures and is not
written back to the OME-Zarr data.

Examples
--------
Choose the model and sample interactively::

    python scripts/denoising/n2v_validation.py

Process every image with the time-lapse model using frame 0::

    python scripts/denoising/n2v_validation.py --model time --scope all --frame 0

Process one WGA-DAPI image with the DAPI model::

    python scripts/denoising/n2v_validation.py --model dapi --scope one \
        --sample WT_CSP_NHS_ROI1_SIM --frame 0
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import tifffile


SCRIPT_FILE = Path(__file__).resolve()


def find_project_root() -> Path:
    """Locate the repository root containing ``scripts`` and ``src/PFT``."""

    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT repository root from this script.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.decoder_omezar import ome_zarr_to_n2v_2d_stack  # noqa: E402
from PFT.core_prog_parts.denoising.n2v_denoising import (  # noqa: E402
    load_n2v_model_by_key,
    predict_yxc,
)
from PFT.core_prog_parts.denoising.n2v_workflow import (  # noqa: E402
    MODEL_SPECS,
    N2VModelSpec,
    choose_model_key_interactive,
    discover_original_samples,
    get_model_spec,
    n2v_results_root,
    output_provenance_attributes,
    sample_name_from_zarr,
    source_array_properties,
    source_scale_for_output_axes,
    utc_now_iso,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr  # noqa: E402


@dataclass(frozen=True)
class InferenceRecord:
    """Record the outcome of one source-image prediction."""

    status: str
    model_key: str
    model_name: str
    dataset: str
    sample: str
    frame_index: int
    source_channels: str
    source_zarr: str
    output_zarr: str
    output_axes: str
    output_shape: str
    output_dtype: str
    input_normalization: str
    output_normalization: str
    issue: str


def choose_from_list(items: Sequence[str], prompt: str) -> int:
    """Return the zero-based index selected from a numbered terminal list."""

    print(prompt)
    for index, item in enumerate(items, start=1):
        print(f"  {index}) {item}")
    while True:
        value = input("Choose number: ").strip()
        try:
            index = int(value)
        except ValueError:
            index = -1
        if 1 <= index <= len(items):
            return index - 1
        print("Invalid selection.")


def load_source_stack(sample_zarr: Path, spec: N2VModelSpec) -> np.ndarray:
    """Load the model-specific original stack as raw-range ``N,Y,X,C`` data."""

    channel_stacks: list[np.ndarray] = []
    for channel in spec.channels:
        stack = np.asarray(
            ome_zarr_to_n2v_2d_stack(
                sample_zarr,
                channel=int(channel),
                time=None,
                z=None,
                normalize=None,
            ),
            dtype=np.float32,
        )
        if stack.ndim != 4 or stack.shape[-1] != 1:
            raise ValueError(f"Expected N,Y,X,1 for channel {channel}; got {stack.shape}.")
        channel_stacks.append(stack)

    frame_count = min(stack.shape[0] for stack in channel_stacks)
    if frame_count < 1:
        raise ValueError("The selected source image contains no two-dimensional frames.")
    if len({stack.shape[0] for stack in channel_stacks}) != 1:
        print(f"[WARN] {sample_name_from_zarr(sample_zarr)}: channel frame counts differ; using {frame_count}.")
    return np.concatenate([stack[:frame_count] for stack in channel_stacks], axis=-1)


def prediction_storage_layout(prediction_yxc: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert one YXC prediction to OME-NGFF ``yx`` or ``cyx`` storage order."""

    prediction = np.asarray(prediction_yxc, dtype=np.float32)
    if prediction.ndim != 3:
        raise ValueError(f"Expected Y,X,C prediction, got {prediction.shape}.")
    if prediction.shape[-1] == 1:
        return prediction[..., 0], "yx"
    return np.transpose(prediction, (2, 0, 1)), "cyx"


def percentile_display_limits(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> tuple[float, float]:
    """Return robust display limits without modifying numeric source data."""

    array = np.asarray(image, dtype=np.float32)
    minimum, maximum = np.percentile(array, (low, high))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
    if maximum <= minimum:
        maximum = minimum + 1.0
    return float(minimum), float(maximum)


def normalize_for_display(image: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Create a clipped [0,1] display array using externally supplied limits."""

    low, high = limits
    return np.clip((np.asarray(image, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


def rgb_from_channels(image_yxc: np.ndarray, limits: Sequence[tuple[float, float]]) -> np.ndarray:
    """Create a display-only RGB image with DAPI/HADA blue and WGA green."""

    image = np.asarray(image_yxc, dtype=np.float32)
    rgb = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    if image.shape[-1] == 1:
        rgb[..., 2] = normalize_for_display(image[..., 0], limits[0])
    else:
        rgb[..., 2] = normalize_for_display(image[..., 0], limits[0])
        rgb[..., 1] = normalize_for_display(image[..., 1], limits[1])
    return rgb


def save_comparison_outputs(
    *,
    raw_yxc: np.ndarray,
    prediction_yxc: np.ndarray,
    output_directory: Path,
    title: str,
) -> None:
    """Save raw/N2V/residual figures and the numerical residual TIFF.

    Raw-derived display limits are also applied to the N2V image.  The absolute
    residual is calculated from unnormalized numeric arrays and saved as
    ``float32``.  These visualization files do not affect the OME-Zarr output.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    limits = [percentile_display_limits(raw_yxc[..., channel]) for channel in range(raw_yxc.shape[-1])]
    raw_rgb = rgb_from_channels(raw_yxc, limits)
    prediction_rgb = rgb_from_channels(prediction_yxc, limits)
    residual_channels = np.abs(prediction_yxc.astype(np.float32) - raw_yxc.astype(np.float32))
    residual = np.mean(residual_channels, axis=-1)
    tifffile.imwrite(output_directory / "absolute_difference_float32.tif", residual.astype(np.float32))

    residual_limit = float(np.percentile(residual, 99.9)) if residual.size else 1.0
    residual_limit = max(residual_limit, 1e-6)
    figure = plt.figure(figsize=(15, 5), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index) for index in (1, 2, 3)]
    axes[0].imshow(raw_rgb)
    axes[0].set_title("RAW")
    axes[1].imshow(prediction_rgb)
    axes[1].set_title("N2V")
    axes[2].imshow(residual, cmap="gray", vmin=0, vmax=residual_limit)
    axes[2].set_title("Absolute difference")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.savefig(output_directory / "raw_n2v_difference.png", dpi=300)
    figure.savefig(output_directory / "raw_n2v_difference.pdf")
    plt.close(figure)

    metrics = {
        "difference_mean": float(np.mean(residual)),
        "difference_standard_deviation": float(np.std(residual)),
        "difference_p99": float(np.percentile(residual, 99.0)),
        "difference_p999": float(np.percentile(residual, 99.9)),
        "difference_maximum": float(np.max(residual)),
    }
    (output_directory / "difference_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def process_sample(
    *,
    sample_zarr: Path,
    spec: N2VModelSpec,
    frame_index: int,
    overwrite: bool,
    save_comparison: bool,
) -> InferenceRecord:
    """Denoise one sample frame, save OME-Zarr provenance, and return a report row."""

    sample = sample_name_from_zarr(sample_zarr)
    output_zarr = (
        n2v_results_root(PROJECT_ROOT)
        / spec.dataset
        / sample
        / spec.variant
        / "denoised"
        / "image.ome.zarr"
    )
    try:
        source_stack = load_source_stack(sample_zarr, spec)
        if not 0 <= frame_index < source_stack.shape[0]:
            raise IndexError(
                f"Frame {frame_index} is outside 0..{source_stack.shape[0] - 1} for {sample}."
            )
        raw_yxc = source_stack[frame_index]
        model = load_n2v_model_by_key(spec.key)
        prediction_yxc = predict_yxc(raw_yxc, model)
        output_array, output_axes = prediction_storage_layout(prediction_yxc)

        source_properties = source_array_properties(sample_zarr)
        provenance = output_provenance_attributes(
            spec=spec,
            source_properties=source_properties,
            source_frame_index=frame_index,
            output_axes=output_axes,
            output_shape=output_array.shape,
            output_dtype=str(output_array.dtype),
        )
        provenance["pft_n2v_output_minimum"] = float(np.min(output_array))
        provenance["pft_n2v_output_maximum"] = float(np.max(output_array))
        provenance["pft_n2v_output_mean"] = float(np.mean(output_array))
        provenance["pft_n2v_output_standard_deviation"] = float(np.std(output_array))

        save_ome_zarr(
            output_zarr,
            output_array,
            axes=output_axes,
            meta=None,
            overwrite=overwrite,
            pyramid_3d=False,
            pyramid_max_layer=0,
            coordinate_scale=source_scale_for_output_axes(source_properties, output_axes),
            extra_attrs=provenance,
        )

        if save_comparison:
            save_comparison_outputs(
                raw_yxc=raw_yxc,
                prediction_yxc=prediction_yxc,
                output_directory=output_zarr.parent.parent / "analysis",
                title=f"{spec.dataset} | {sample} | {spec.variant} | frame {frame_index}",
            )

        print(f"[PASS] {spec.dataset} | {sample} | frame {frame_index} -> {output_zarr}")
        return InferenceRecord(
            status="PASS",
            model_key=spec.key,
            model_name=spec.model_name,
            dataset=spec.dataset,
            sample=sample,
            frame_index=frame_index,
            source_channels=",".join(str(value) for value in spec.channels),
            source_zarr=str(sample_zarr.resolve()),
            output_zarr=str(output_zarr.resolve()),
            output_axes=output_axes,
            output_shape=str(tuple(int(value) for value in output_array.shape)),
            output_dtype=str(output_array.dtype),
            input_normalization="none",
            output_normalization="none",
            issue="",
        )
    except Exception as exc:
        print(f"[FAIL] {spec.dataset} | {sample}: {type(exc).__name__}: {exc}")
        return InferenceRecord(
            status="FAIL",
            model_key=spec.key,
            model_name=spec.model_name,
            dataset=spec.dataset,
            sample=sample,
            frame_index=frame_index,
            source_channels=",".join(str(value) for value in spec.channels),
            source_zarr=str(sample_zarr.resolve()),
            output_zarr=str(output_zarr.resolve()),
            output_axes="",
            output_shape="",
            output_dtype="",
            input_normalization="none",
            output_normalization="none",
            issue=f"{type(exc).__name__}: {exc}",
        )


def save_run_report(records: Sequence[InferenceRecord], spec: N2VModelSpec) -> Path:
    """Save CSV, JSON, and TXT summaries for one inference command."""

    report_dir = n2v_results_root(PROJECT_ROOT) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_iso().replace(":", "-").replace("+", "_")
    stem = f"n2v_inference_{spec.key}_{timestamp}"

    rows = [asdict(record) for record in records]
    csv_path = report_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    passed = sum(record.status == "PASS" for record in records)
    failed = len(records) - passed
    lines = [
        "PFT 2D Noise2Void inference report",
        "=" * 72,
        f"Generated (UTC): {utc_now_iso()}",
        f"Model key: {spec.key}",
        f"Model name: {spec.model_name}",
        f"Dataset: {spec.dataset}",
        f"Source root: {PROJECT_ROOT / 'results' / 'img' / spec.dataset}",
        f"Input normalization: none",
        f"Output normalization: none",
        f"Records: {len(records)}",
        f"PASS: {passed}",
        f"FAIL: {failed}",
        "",
    ]
    for record in records:
        lines.append(
            f"{record.status} | {record.dataset} | {record.sample} | frame {record.frame_index}"
            + (f" | {record.issue}" if record.issue else "")
        )
    txt_path = report_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for one/all sample inference."""

    parser = argparse.ArgumentParser(
        description="Run one trained 2D N2V model on one or all original OME-Zarr images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), help="N2V model to apply.")
    parser.add_argument("--scope", choices=("one", "all"), help="Process one selected sample or all samples for the model dataset.")
    parser.add_argument("--sample", help="Sample directory name for --scope one. Omit for an interactive list.")
    parser.add_argument("--frame", type=int, default=0, help="Original frame index paired with the N2V output.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing N2V OME-Zarr output.")
    parser.add_argument("--no-comparison", action="store_true", help="Do not create raw/N2V/difference PNG and PDF figures.")
    return parser


def resolve_samples(
    *, spec: N2VModelSpec, scope: str | None, sample_name: str | None
) -> list[Path]:
    """Resolve one or all source samples, using terminal menus when needed."""

    samples = discover_original_samples(spec, PROJECT_ROOT)
    if not samples:
        raise FileNotFoundError(
            f"No image.ome.zarr files found below {PROJECT_ROOT / 'results' / 'img' / spec.dataset}."
        )
    selected_scope = scope
    if selected_scope is None:
        selected_scope = ("one", "all")[choose_from_list(
            ("Process one image", "Process all images"),
            "Choose inference scope:",
        )]
    if selected_scope == "all":
        return samples

    by_name = {sample_name_from_zarr(path): path for path in samples}
    if sample_name is not None:
        try:
            return [by_name[sample_name]]
        except KeyError as exc:
            available = ", ".join(sorted(by_name))
            raise ValueError(f"Unknown sample {sample_name!r}. Available samples: {available}") from exc
    names = sorted(by_name)
    selected = names[choose_from_list(names, "Choose one source image:")]
    return [by_name[selected]]


def main(argv: list[str] | None = None) -> int:
    """Run model selection, source selection, inference, and report generation."""

    args = build_parser().parse_args(argv)
    model_key = args.model or choose_model_key_interactive("Choose the N2V model for inference")
    spec = get_model_spec(model_key)
    samples = resolve_samples(spec=spec, scope=args.scope, sample_name=args.sample)

    records = [
        process_sample(
            sample_zarr=sample,
            spec=spec,
            frame_index=args.frame,
            overwrite=args.overwrite,
            save_comparison=not args.no_comparison,
        )
        for sample in samples
    ]
    report_path = save_run_report(records, spec)
    print(f"\nInference report: {report_path}")
    return 0 if all(record.status == "PASS" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
