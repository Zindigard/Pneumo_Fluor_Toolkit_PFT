"""Screen pretrained Cellpose-SAM or Omnipose models without training.

This script is intended for the first instance-segmentation experiment. It runs
an inference-parameter grid on prepared ``segmentation_input.ome.zarr`` files
and compares the predicted foreground union against the existing binary U-Net
foreground mask.

The U-Net mask is valid for semantic foreground metrics:

* binary Dice;
* binary intersection over union (IoU);
* predicted/reference foreground-area ratio.

It is not an instance-labelled reference. Therefore, this screening does not
calculate instance F1, split/merge errors, or one-to-one cell matching. Manual
instance masks remain necessary only for final instance-level validation or
model fine-tuning.

Outputs
-------

Results are stored below::

    results/segmentation_screening/<family>/<dataset>/<source_mode>/
        <model>/<run_name>/
            screening_metrics.csv
            best_parameters.json
            screening_report.html
            parameter_###/<sample>/comparison.png

Cellpose example::

    python scripts/segmentation/screen_pretrained_segmentation.py `
        --family cellpose `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --model cpsam_v2 `
        --mode one `
        --flow-thresholds 0.2 0.4 0.6 `
        --cellprob-thresholds -1 0 1

Omnipose example::

    python scripts/segmentation/screen_pretrained_segmentation.py `
        --family omnipose `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --model-name bact_fluor_omni `
        --mode one `
        --flow-thresholds 0 0.4 0.8 `
        --mask-thresholds -1 0 1
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence

import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr  # noqa: E402
from PFT.core_prog_parts.segmentation.instance_segmentation_core import (  # noqa: E402
    PredictionConfig,
    list_prepared_inputs,
    load_prediction_model,
    load_prepared_image,
    predict_one,
    semantic_dice,
    semantic_iou,
)
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
)

SCREENING_FAMILIES = ("cellpose", "omnipose")


def _choose(title: str, values: Sequence[str]) -> str:
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _parse_diameters(values: Sequence[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        parsed.append(None if value.lower() in {"none", "auto"} else float(value))
    return tuple(parsed)


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _rebase_recorded_path(value: str | Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.exists():
        return candidate.resolve()

    windows_parts = list(PureWindowsPath(str(value)).parts)
    lower = [part.lower() for part in windows_parts]
    for anchor in ("results", "models", "scripts", "src"):
        if anchor in lower:
            index = lower.index(anchor)
            rebased = PROJECT_ROOT.joinpath(*windows_parts[index:])
            if rebased.exists():
                return rebased.resolve()
    return candidate


def _root_processing(path: Path) -> dict[str, Any]:
    try:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        processing = dict(group.attrs.asdict()).get("pft_processing", {})
        return dict(processing) if isinstance(processing, dict) else {}
    except Exception:
        return {}


def _remove_axis(array: np.ndarray, axes: str, axis: str, index: int = 0) -> tuple[np.ndarray, str]:
    position = axes.index(axis)
    array = np.take(array, index, axis=position)
    axes = axes[:position] + axes[position + 1 :]
    return array, axes


def _load_unet_binary_reference(input_zarr: Path, expected_yx: tuple[int, int]) -> tuple[np.ndarray, Path]:
    report_path = input_zarr.parent / "segmentation_input_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing segmentation input report required to resolve the U-Net mask: {report_path}"
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    processing = payload.get("processing", {})
    record = payload.get("record", {})
    mask_value = processing.get("predicted_mask_omezarr") or record.get("mask_zarr")
    if not mask_value:
        raise RuntimeError(
            "This prepared input has no associated U-Net mask. Binary U-Net reference "
            "screening is unavailable for unmasked source modes."
        )
    mask_path = _rebase_recorded_path(mask_value)
    if not mask_path.is_dir():
        raise FileNotFoundError(mask_path)

    array, axes = load_ome_zarr(mask_path, level=0, as_numpy=True)
    mask = np.asarray(array)
    axes = str(axes).lower()

    for axis in ("t", "c"):
        if axis in axes:
            size = int(mask.shape[axes.index(axis)])
            if size != 1:
                raise ValueError(
                    f"Binary U-Net reference must have singleton {axis.upper()} axis, "
                    f"received {size}: {mask_path}"
                )
            mask, axes = _remove_axis(mask, axes, axis)

    if "z" in axes:
        source_value = record.get("source_zarr") or processing.get("source_omezarr")
        if not source_value:
            raise RuntimeError("A Z mask requires source metadata with the selected MIP slice")
        source_path = _rebase_recorded_path(source_value)
        source_processing = _root_processing(source_path)
        target_slice = source_processing.get("target_slice_1based_for_qc_and_mask")
        if target_slice is None:
            raise RuntimeError(
                f"A Z mask was found but no target slice is recorded in {source_path}"
            )
        z_index = int(target_slice) - 1
        z_size = int(mask.shape[axes.index("z")])
        if not 0 <= z_index < z_size:
            raise IndexError(f"Target slice {target_slice} is outside Z={z_size}")
        mask, axes = _remove_axis(mask, axes, "z", z_index)

    if axes != "yx":
        raise ValueError(f"Expected final binary reference axes YX, received {axes!r}: {mask_path}")
    if tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(
            f"U-Net reference shape {mask.shape} does not match input YX {expected_yx}: {mask_path}"
        )
    binary = np.asarray(mask) > 0
    if np.count_nonzero(binary) == 0:
        raise ValueError(f"U-Net reference mask is empty: {mask_path}")
    return binary, mask_path


def _parameter_grid(
    family: str,
    diameters: tuple[float | None, ...],
    flow_thresholds: Sequence[float],
    cellprob_thresholds: Sequence[float],
    mask_thresholds: Sequence[float],
    min_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    if family == "cellpose":
        return [
            {
                "diameter": diameter,
                "flow_threshold": flow,
                "cellprob_threshold": cellprob,
                "min_size": min_size,
            }
            for diameter, flow, cellprob, min_size in product(
                diameters, flow_thresholds, cellprob_thresholds, min_sizes
            )
        ]
    return [
        {
            "diameter": diameter,
            "flow_threshold": flow,
            "mask_threshold": mask_threshold,
            "min_size": min_size,
        }
        for diameter, flow, mask_threshold, min_size in product(
            diameters, flow_thresholds, mask_thresholds, min_sizes
        )
    ]


def _input_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 2:
        gray = np.round(array * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected YX or YXC image, received {array.shape}")
    if array.shape[-1] == 1:
        gray = np.round(array[..., 0] * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    if array.shape[-1] == 2:
        blue = np.round(array[..., 0] * 255.0).astype(np.uint8)
        green = np.round(array[..., 1] * 255.0).astype(np.uint8)
        red = np.zeros_like(blue)
        return np.stack([red, green, blue], axis=-1)
    return np.round(array[..., :3] * 255.0).astype(np.uint8)


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    interior = binary.copy()
    interior[1:, :] &= binary[:-1, :]
    interior[:-1, :] &= binary[1:, :]
    interior[:, 1:] &= binary[:, :-1]
    interior[:, :-1] &= binary[:, 1:]
    return binary & ~interior


def _labels_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    values = labels[positive]
    output[..., 0][positive] = ((values * 67 + 31) % 205 + 50).astype(np.uint8)
    output[..., 1][positive] = ((values * 97 + 73) % 205 + 50).astype(np.uint8)
    output[..., 2][positive] = ((values * 131 + 17) % 205 + 50).astype(np.uint8)
    return output


def _fit_preview(image: Image.Image, max_size: int, *, nearest: bool = False) -> Image.Image:
    if max(image.size) <= max_size:
        return image
    scale = max_size / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize(size, resampling)


def _panel(rgb: np.ndarray, title: str, max_size: int, *, nearest: bool = False) -> Image.Image:
    image = _fit_preview(Image.fromarray(rgb, mode="RGB"), max_size, nearest=nearest)
    canvas = Image.new("RGB", (image.width, image.height + 32), "white")
    canvas.paste(image, (0, 32))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    return canvas


def _save_comparison(
    path: Path,
    image: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, Any],
    *,
    max_size: int,
) -> None:
    input_rgb = _input_rgb(image)
    ref_rgb = np.zeros_like(input_rgb)
    ref_rgb[..., 1] = np.where(reference, 255, 0).astype(np.uint8)
    pred_rgb = _labels_rgb(prediction)

    overlay = input_rgb.astype(np.float32)
    ref_boundary = _boundary(reference)
    pred_boundary = _boundary(prediction > 0)
    overlay[ref_boundary] = np.array([0, 255, 0], dtype=np.float32)
    overlay[pred_boundary] = np.array([255, 0, 255], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    titles = [
        "Prepared input",
        "U-Net binary reference",
        f"Prediction: {metrics['predicted_instances']} instances",
        (
            f"Overlay: IoU={metrics['semantic_iou']:.4f}, "
            f"Dice={metrics['semantic_dice']:.4f}"
        ),
    ]
    panels = [
        _panel(input_rgb, titles[0], max_size),
        _panel(ref_rgb, titles[1], max_size, nearest=True),
        _panel(pred_rgb, titles[2], max_size, nearest=True),
        _panel(overlay, titles[3], max_size),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _approx_component_count(binary: np.ndarray) -> int | None:
    try:
        from scipy.ndimage import label

        _labels, count = label(np.asarray(binary) > 0)
        return int(count)
    except Exception:
        return None


def _write_html(path: Path, rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    sample_rows = [row for row in rows if row["sample_key"] != "__MEAN__"]
    sections = []
    for row in sample_rows:
        preview = Path(str(row["comparison_png"]))
        try:
            relative = preview.relative_to(path.parent).as_posix()
        except ValueError:
            relative = preview.as_uri()
        parameters = {
            key: value
            for key, value in row.items()
            if key in {"diameter", "flow_threshold", "cellprob_threshold", "mask_threshold", "min_size"}
            and value not in {"", None}
        }
        sections.append(
            "<section>"
            f"<h3>Parameter {row['parameter_index']} | {html.escape(str(row['sample_key']))}</h3>"
            f"<p><code>{html.escape(json.dumps(parameters))}</code><br>"
            f"IoU={row['semantic_iou']:.4f}, Dice={row['semantic_dice']:.4f}, "
            f"instances={row['predicted_instances']}</p>"
            f"<img src=\"{html.escape(relative)}\" style=\"max-width:100%;height:auto\">"
            "</section>"
        )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PFT pretrained segmentation screening</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} section{{margin:28px 0;padding-top:12px;border-top:1px solid #ccc}} code{{white-space:pre-wrap}}</style>
</head><body>
<h1>PFT pretrained segmentation screening</h1>
<h2>Best mean parameters</h2>
<pre>{html.escape(json.dumps(best, indent=2, default=str))}</pre>
<p>Green outline: U-Net binary reference. Magenta outline: model prediction.</p>
{''.join(sections)}
</body></html>"""
    path.write_text(body, encoding="utf-8")


def main(default_family: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Screen pretrained Cellpose-SAM or Omnipose parameter grids against "
            "binary U-Net foreground masks without model training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family",
        choices=(default_family,) if default_family else SCREENING_FAMILIES,
        default=default_family,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--mode", choices=("one", "all"), default=None)
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--diameters", nargs="+", default=["none"])
    parser.add_argument("--flow-thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--cellprob-thresholds", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--mask-thresholds", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--min-sizes", type=int, nargs="+", default=[15])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-preview-size", type=int, default=768)
    parser.add_argument("--save-labels", action="store_true")
    args = parser.parse_args()

    family = args.family or _choose("Choose pretrained model family", SCREENING_FAMILIES)
    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is invalid for {dataset!r}: {valid_modes}")

    items = list_prepared_inputs(PROJECT_ROOT, dataset, source_mode)
    if not items:
        raise FileNotFoundError(
            "No prepared segmentation inputs were found. Run prepare_segmentation_inputs.py first."
        )

    selected_keys = set(args.sample or [])
    run_mode = args.mode or _choose("Screen one sample or all prepared samples", ("one", "all"))
    if run_mode == "one" and not selected_keys:
        selected_keys = {_choose("Choose prepared sample", [item.sample_key for item in items])}
    if selected_keys:
        items = [item for item in items if item.sample_key in selected_keys]
    if not items:
        raise RuntimeError("No prepared samples matched the requested selection")

    diameters = _parse_diameters(args.diameters)
    grid = _parameter_grid(
        family,
        diameters,
        args.flow_thresholds,
        args.cellprob_thresholds,
        args.mask_thresholds,
        args.min_sizes,
    )

    base_cfg = PredictionConfig(
        project_root=PROJECT_ROOT,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        model=args.model,
        model_name=args.model_name,
        gpu=not args.cpu,
        batch_size=args.batch_size,
    )
    model = load_prediction_model(base_cfg)
    model_display = (
        args.model_name
        or (Path(str(args.model)).stem if args.model else None)
        or ("cpsam_v2" if family == "cellpose" else "bact_fluor_omni")
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        PROJECT_ROOT
        / "results"
        / "segmentation_screening"
        / family
        / dataset
        / source_mode
        / model_display
        / run_name
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("\nPFT pretrained segmentation screening")
    print("=" * 72)
    print(f"Family:             {family}")
    print(f"Model:              {model_display}")
    print(f"Dataset:            {dataset}")
    print(f"Source mode:        {source_mode}")
    print(f"Samples:            {len(items)}")
    print(f"Parameter settings: {len(grid)}")
    print("Reference:          binary U-Net mask")
    print("Ranking:            mean semantic IoU, then mean Dice")
    print("Instance F1:        not calculated without instance labels")
    print("Model normalization: DISABLED")

    loaded: list[tuple[Any, np.ndarray, np.ndarray, Path]] = []
    for item in items:
        image = load_prepared_image(item.input_zarr)
        reference, reference_path = _load_unet_binary_reference(
            item.input_zarr, tuple(int(value) for value in image.shape[:2])
        )
        loaded.append((item, image, reference, reference_path))

    rows: list[dict[str, Any]] = []
    for parameter_index, parameters in enumerate(grid, start=1):
        print(f"\n[{parameter_index:03d}/{len(grid):03d}] {parameters}")
        parameter_rows: list[dict[str, Any]] = []
        for item, image, reference, reference_path in loaded:
            prediction_cfg = PredictionConfig(**asdict(base_cfg))
            for key, value in parameters.items():
                setattr(prediction_cfg, key, value)
            prediction = predict_one(model, image, prediction_cfg)
            predicted_binary = prediction > 0
            ref_pixels = int(np.count_nonzero(reference))
            pred_pixels = int(np.count_nonzero(predicted_binary))
            sample_dir = output_root / f"parameter_{parameter_index:03d}" / Path(item.sample_key)
            comparison_path = sample_dir / "comparison.png"
            row: dict[str, Any] = {
                "parameter_index": parameter_index,
                "sample_key": item.sample_key,
                **parameters,
                "semantic_dice": semantic_dice(reference, predicted_binary),
                "semantic_iou": semantic_iou(reference, predicted_binary),
                "reference_foreground_pixels": ref_pixels,
                "prediction_foreground_pixels": pred_pixels,
                "foreground_area_ratio": float(pred_pixels / ref_pixels) if ref_pixels else float("nan"),
                "predicted_instances": int(np.unique(prediction[prediction > 0]).size),
                "approx_reference_components": _approx_component_count(reference),
                "reference_mask": str(reference_path),
                "comparison_png": str(comparison_path),
            }
            _save_comparison(
                comparison_path,
                image,
                reference,
                prediction,
                row,
                max_size=args.max_preview_size,
            )
            if args.save_labels:
                sample_dir.mkdir(parents=True, exist_ok=True)
                max_label = int(np.max(prediction)) if prediction.size else 0
                dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
                tiff.imwrite(sample_dir / "predicted_labels.tif", prediction.astype(dtype))
            parameter_rows.append(row)
            rows.append(row)
            print(
                f"  {item.sample_key}: IoU={row['semantic_iou']:.4f}, "
                f"Dice={row['semantic_dice']:.4f}, instances={row['predicted_instances']}"
            )

        aggregate: dict[str, Any] = {
            "parameter_index": parameter_index,
            "sample_key": "__MEAN__",
            **parameters,
            "semantic_dice": float(np.mean([row["semantic_dice"] for row in parameter_rows])),
            "semantic_iou": float(np.mean([row["semantic_iou"] for row in parameter_rows])),
            "reference_foreground_pixels": int(
                sum(row["reference_foreground_pixels"] for row in parameter_rows)
            ),
            "prediction_foreground_pixels": int(
                sum(row["prediction_foreground_pixels"] for row in parameter_rows)
            ),
            "foreground_area_ratio": float(
                np.mean([row["foreground_area_ratio"] for row in parameter_rows])
            ),
            "predicted_instances": float(
                np.mean([row["predicted_instances"] for row in parameter_rows])
            ),
            "approx_reference_components": float(
                np.mean(
                    [
                        row["approx_reference_components"]
                        for row in parameter_rows
                        if row["approx_reference_components"] is not None
                    ]
                )
            )
            if any(row["approx_reference_components"] is not None for row in parameter_rows)
            else None,
            "reference_mask": "",
            "comparison_png": "",
        }
        rows.append(aggregate)

    aggregate_rows = [row for row in rows if row["sample_key"] == "__MEAN__"]
    best = max(
        aggregate_rows,
        key=lambda row: (row["semantic_iou"], row["semantic_dice"]),
    )

    csv_path = output_root / "screening_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "model": model_display,
        "dataset": dataset,
        "source_mode": source_mode,
        "reference_type": "binary_unet_foreground",
        "valid_metrics": ["semantic_iou", "semantic_dice", "foreground_area_ratio"],
        "invalid_without_instance_labels": [
            "instance_f1",
            "split_error",
            "merge_error",
            "one_to_one_cell_matching",
        ],
        "best_parameters_and_metrics": best,
        "evaluated_samples": [item.sample_key for item, *_ in loaded],
        "model_normalization": False,
        "metrics_csv": str(csv_path),
    }
    best_path = output_root / "best_parameters.json"
    best_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    html_path = output_root / "screening_report.html"
    _write_html(html_path, rows, best)

    print("\nBest mean semantic setting")
    print(json.dumps(best, indent=2, default=str))
    print(f"\nMetrics: {csv_path}")
    print(f"Report:  {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
