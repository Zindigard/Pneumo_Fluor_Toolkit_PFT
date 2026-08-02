"""Screen pretrained instance-segmentation models against manual instance masks.

The script runs Cellpose, Omnipose, or StarDist without fine-tuning and compares
predicted integer instance labels directly with manually annotated integer masks.

Default screening behaviour
---------------------------
If no ``--sample`` is supplied, the script randomly selects exactly two valid
annotated image/mask pairs. Selection is reproducible through ``--seed``.

Metrics
-------
Pixel-level foreground metrics:

* semantic IoU;
* semantic Dice;
* foreground precision and recall;
* prediction/reference foreground-area ratio.

One-to-one instance metrics at IoU 0.50 and 0.75:

* true positives, false positives, and false negatives;
* instance precision, recall, F1, and Jaccard score;
* mean IoU of matched instances.

Canonical manual masks
----------------------
Full-image labels::

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/mask.tif

Crop labels::

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/
        crops/<train|validation>/<crop_id>/mask.tif
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
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

from PFT.core_prog_parts.segmentation.instance_segmentation_core import (  # noqa: E402
    ANNOTATION_SOURCES,
    ANNOTATION_SPLITS,
    PredictionConfig,
    list_training_inputs,
    load_instance_mask,
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

SCREENING_FAMILIES = ("cellpose", "omnipose", "stardist")


def _choose(title: str, values: Sequence[str]) -> str:
    if not values:
        raise ValueError(f"No values are available for: {title}")
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _parse_optional_floats(values: Sequence[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        parsed.append(None if value.lower() in {"none", "default", "auto"} else float(value))
    return tuple(parsed)


def _parse_diameters(values: Sequence[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        parsed.append(None if value.lower() in {"none", "auto"} else float(value))
    return tuple(parsed)


def _parameter_grid(
    family: str,
    diameters: tuple[float | None, ...],
    flow_thresholds: Sequence[float],
    cellprob_thresholds: Sequence[float],
    mask_thresholds: Sequence[float],
    prob_thresholds: tuple[float | None, ...],
    nms_thresholds: tuple[float | None, ...],
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
    if family == "omnipose":
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
    return [
        {"prob_thresh": prob_thresh, "nms_thresh": nms_thresh}
        for prob_thresh, nms_thresh in product(prob_thresholds, nms_thresholds)
    ]


def _source_sample_key(sample_key: str) -> str:
    return sample_key.replace("\\", "/").split("/crops/", 1)[0]


def _load_valid_pair(item: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    image = load_prepared_image(item.input_zarr)
    reference = load_instance_mask(item.training_mask, tuple(int(v) for v in image.shape[:2]))
    return item, image, reference


def _select_pairs(
    candidates: Sequence[Any],
    *,
    selected_keys: set[str],
    sample_count: int,
    seed: int,
    distinct_sources: bool,
) -> tuple[list[tuple[Any, np.ndarray, np.ndarray]], list[dict[str, str]]]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    candidate_map = {item.sample_key: item for item in candidates}
    invalid: list[dict[str, str]] = []

    if selected_keys:
        unknown = sorted(selected_keys - set(candidate_map))
        if unknown:
            raise RuntimeError(
                "Requested annotated samples were not found: " + ", ".join(unknown)
            )
        ordered = [candidate_map[key] for key in sorted(selected_keys)]
    else:
        ordered = list(candidates)
        random.Random(seed).shuffle(ordered)

    selected: list[tuple[Any, np.ndarray, np.ndarray]] = []
    selected_sources: set[str] = set()

    for item in ordered:
        if not item.training_mask.is_file():
            continue
        source_key = _source_sample_key(item.sample_key)
        if distinct_sources and source_key in selected_sources:
            continue
        try:
            loaded = _load_valid_pair(item)
        except Exception as exc:
            invalid.append({"sample_key": item.sample_key, "error": str(exc)})
            continue
        selected.append(loaded)
        selected_sources.add(source_key)
        if not selected_keys and len(selected) >= sample_count:
            break

    required = len(selected_keys) if selected_keys else sample_count
    if len(selected) < required:
        details = ""
        if invalid:
            details = " Invalid pairs: " + "; ".join(
                f"{row['sample_key']}: {row['error']}" for row in invalid[:5]
            )
        raise RuntimeError(
            f"Only {len(selected)} valid annotated pairs were available, but {required} "
            f"were requested.{details}"
        )
    return selected, invalid


def _contingency_iou_matrix(
    reference: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    ref_ids = np.unique(ref[ref > 0])
    pred_ids = np.unique(pred[pred > 0])

    if ref_ids.size == 0 or pred_ids.size == 0:
        return (
            np.zeros((ref_ids.size, pred_ids.size), dtype=np.float64),
            ref_ids,
            pred_ids,
        )

    ref_index = np.searchsorted(ref_ids, ref)
    pred_index = np.searchsorted(pred_ids, pred)
    both = (ref > 0) & (pred > 0)

    combined = ref_index[both] * pred_ids.size + pred_index[both]
    intersections = np.bincount(
        combined,
        minlength=int(ref_ids.size * pred_ids.size),
    ).reshape(ref_ids.size, pred_ids.size)

    ref_areas = np.array([np.count_nonzero(ref == label) for label in ref_ids], dtype=np.int64)
    pred_areas = np.array([np.count_nonzero(pred == label) for label in pred_ids], dtype=np.int64)
    unions = ref_areas[:, None] + pred_areas[None, :] - intersections
    matrix = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return matrix, ref_ids, pred_ids


def _assignment_pairs(iou_matrix: np.ndarray) -> list[tuple[int, int, float]]:
    if iou_matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-iou_matrix)
        return [(int(r), int(c), float(iou_matrix[r, c])) for r, c in zip(rows, cols)]
    except Exception:
        candidates = sorted(
            (
                (float(iou_matrix[r, c]), r, c)
                for r in range(iou_matrix.shape[0])
                for c in range(iou_matrix.shape[1])
                if iou_matrix[r, c] > 0
            ),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for value, row, col in candidates:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((row, col, value))
        return pairs


def _instance_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.50, 0.75),
) -> dict[str, Any]:
    matrix, ref_ids, pred_ids = _contingency_iou_matrix(reference, prediction)
    pairs = _assignment_pairs(matrix)
    metrics: dict[str, Any] = {
        "reference_instances": int(ref_ids.size),
        "predicted_instances": int(pred_ids.size),
        "mean_best_reference_iou": (
            float(np.mean(np.max(matrix, axis=1))) if ref_ids.size and pred_ids.size else 0.0
        ),
        "mean_best_prediction_iou": (
            float(np.mean(np.max(matrix, axis=0))) if ref_ids.size and pred_ids.size else 0.0
        ),
    }

    for threshold in thresholds:
        suffix = str(int(round(threshold * 100)))
        matched_ious = [value for _row, _col, value in pairs if value >= threshold]
        true_positive = len(matched_ious)
        false_positive = int(pred_ids.size) - true_positive
        false_negative = int(ref_ids.size) - true_positive
        precision = (
            float(true_positive / (true_positive + false_positive))
            if true_positive + false_positive
            else (1.0 if ref_ids.size == 0 else 0.0)
        )
        recall = (
            float(true_positive / (true_positive + false_negative))
            if true_positive + false_negative
            else 1.0
        )
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        jaccard = (
            float(true_positive / (true_positive + false_positive + false_negative))
            if true_positive + false_positive + false_negative
            else 1.0
        )
        metrics.update(
            {
                f"instance_tp_{suffix}": true_positive,
                f"instance_fp_{suffix}": false_positive,
                f"instance_fn_{suffix}": false_negative,
                f"instance_precision_{suffix}": precision,
                f"instance_recall_{suffix}": recall,
                f"instance_f1_{suffix}": f1,
                f"instance_jaccard_{suffix}": jaccard,
                f"mean_matched_iou_{suffix}": (
                    float(np.mean(matched_ious)) if matched_ious else 0.0
                ),
            }
        )
    return metrics


def _semantic_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    ref = np.asarray(reference) > 0
    pred = np.asarray(prediction) > 0
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    ref_pixels = int(np.count_nonzero(ref))
    pred_pixels = int(np.count_nonzero(pred))
    precision = float(tp / (tp + fp)) if tp + fp else (1.0 if ref_pixels == 0 else 0.0)
    recall = float(tp / (tp + fn)) if tp + fn else 1.0
    return {
        "semantic_iou": semantic_iou(reference, prediction),
        "semantic_dice": semantic_dice(reference, prediction),
        "semantic_precision": precision,
        "semantic_recall": recall,
        "reference_foreground_pixels": ref_pixels,
        "prediction_foreground_pixels": pred_pixels,
        "foreground_area_ratio": float(pred_pixels / ref_pixels) if ref_pixels else float("nan"),
    }


def _input_rgb(image: np.ndarray, dataset: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    rgb = np.zeros((*array.shape[:2], 3), dtype=np.float32)
    if array.ndim == 2:
        rgb[..., 2] = array
    elif dataset == "2d_wga_dapi" and array.shape[-1] >= 2:
        rgb[..., 2] = array[..., 0]
        rgb[..., 1] = array[..., 1]
    else:
        rgb[..., 2] = array[..., 0]
        if array.shape[-1] > 1:
            rgb[..., 1] = array[..., 1]
        if array.shape[-1] > 2:
            rgb[..., 0] = array[..., 2]
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


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
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    ImageDraw.Draw(canvas).text((8, 9), title, fill="black")
    return canvas


def _save_comparison(
    path: Path,
    image: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, Any],
    *,
    dataset: str,
    max_size: int,
) -> None:
    input_rgb = _input_rgb(image, dataset)
    ref_rgb = _labels_rgb(reference)
    pred_rgb = _labels_rgb(prediction)

    overlay = input_rgb.astype(np.float32)
    overlay[_boundary(reference)] = np.array([0, 255, 0], dtype=np.float32)
    overlay[_boundary(prediction)] = np.array([255, 0, 255], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    titles = [
        "Prepared input",
        f"Manual labels: {metrics['reference_instances']} instances",
        f"StarDist prediction: {metrics['predicted_instances']} instances",
        (
            f"IoU={metrics['semantic_iou']:.3f}, Dice={metrics['semantic_dice']:.3f}, "
            f"F1@0.50={metrics['instance_f1_50']:.3f}"
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


def _mean_numeric(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.nanmean(values)) if values else float("nan")


def _write_html(path: Path, rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    sample_rows = [row for row in rows if row["sample_key"] != "__MEAN__"]
    sections: list[str] = []
    for row in sample_rows:
        preview = Path(str(row["comparison_png"]))
        try:
            relative = preview.relative_to(path.parent).as_posix()
        except ValueError:
            relative = preview.as_uri()
        parameters = {
            key: row.get(key)
            for key in (
                "diameter",
                "flow_threshold",
                "cellprob_threshold",
                "mask_threshold",
                "prob_thresh",
                "nms_thresh",
                "min_size",
            )
            if row.get(key) not in {"", None}
        }
        sections.append(
            "<section>"
            f"<h3>Parameter {row['parameter_index']} | {html.escape(str(row['sample_key']))}</h3>"
            f"<p><code>{html.escape(json.dumps(parameters))}</code><br>"
            f"Semantic IoU={row['semantic_iou']:.4f}, Dice={row['semantic_dice']:.4f}<br>"
            f"Instance F1@0.50={row['instance_f1_50']:.4f}, "
            f"precision={row['instance_precision_50']:.4f}, "
            f"recall={row['instance_recall_50']:.4f}<br>"
            f"Manual instances={row['reference_instances']}, "
            f"predicted instances={row['predicted_instances']}</p>"
            f"<img src=\"{html.escape(relative)}\" style=\"max-width:100%;height:auto\">"
            "</section>"
        )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PFT manual-mask screening</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} section{{margin:28px 0;padding-top:12px;border-top:1px solid #ccc}} code{{white-space:pre-wrap}}</style>
</head><body>
<h1>PFT pretrained-model screening against manual instance masks</h1>
<h2>Best mean setting</h2>
<pre>{html.escape(json.dumps(best, indent=2, default=str))}</pre>
<p>Green outline: manual annotation. Magenta outline: model prediction.</p>
{''.join(sections)}
</body></html>"""
    path.write_text(body, encoding="utf-8")


def main(default_family: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Screen pretrained Cellpose, Omnipose, or StarDist models directly "
            "against manual integer instance masks."
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
    parser.add_argument(
        "--annotation-source",
        choices=ANNOTATION_SOURCES,
        default="full-images-only",
        help="Which manual annotations may be selected.",
    )
    parser.add_argument(
        "--annotation-split",
        choices=ANNOTATION_SPLITS,
        default="any",
        help="Optional crop split filter. Full-image masks have no explicit split.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Exact annotated sample key. Repeat to evaluate several explicit samples.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=2,
        help="Number of valid annotated pairs randomly selected when --sample is absent.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow-same-source",
        action="store_true",
        help="Allow two crop pairs from the same original source image.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--diameters", nargs="+", default=["none"])
    parser.add_argument("--flow-thresholds", type=float, nargs="+", default=[0.4])
    parser.add_argument("--cellprob-thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument("--mask-thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--prob-thresholds",
        nargs="+",
        default=["none"],
        help="StarDist probability thresholds. 'none' uses the model's stored threshold.",
    )
    parser.add_argument(
        "--nms-thresholds",
        nargs="+",
        default=["none"],
        help="StarDist NMS thresholds. 'none' uses the model's stored threshold.",
    )
    parser.add_argument("--min-sizes", type=int, nargs="+", default=[15])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-preview-size", type=int, default=768)
    parser.add_argument("--save-labels", action="store_true")
    args = parser.parse_args()

    family = args.family or _choose("Choose pretrained model family", SCREENING_FAMILIES)
    dataset = args.dataset or _choose("Choose dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is invalid for {dataset!r}: {valid_modes}")

    candidates = list_training_inputs(
        PROJECT_ROOT,
        dataset,
        source_mode,
        annotation_source=args.annotation_source,
        annotation_split=args.annotation_split,
    )
    candidates = [item for item in candidates if item.training_mask.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No manual image/mask pairs were found for the requested dataset, source mode, "
            "annotation source, and annotation split."
        )

    selected_keys = set(args.sample or [])
    loaded, invalid_pairs = _select_pairs(
        candidates,
        selected_keys=selected_keys,
        sample_count=args.sample_count,
        seed=args.seed,
        distinct_sources=not args.allow_same_source,
    )

    diameters = _parse_diameters(args.diameters)
    prob_thresholds = _parse_optional_floats(args.prob_thresholds)
    nms_thresholds = _parse_optional_floats(args.nms_thresholds)
    grid = _parameter_grid(
        family,
        diameters,
        args.flow_thresholds,
        args.cellprob_thresholds,
        args.mask_thresholds,
        prob_thresholds,
        nms_thresholds,
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
        or {
            "cellpose": "cpsam_v2",
            "omnipose": "bact_fluor_omni",
            "stardist": "2D_versatile_fluo",
        }[family]
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

    selected_names = [item.sample_key for item, _image, _reference in loaded]
    print("\nPFT pretrained segmentation screening")
    print("=" * 78)
    print(f"Family:              {family}")
    print(f"Model:               {model_display}")
    print(f"Dataset:             {dataset}")
    print(f"Source mode:         {source_mode}")
    print(f"Annotation source:   {args.annotation_source}")
    print(f"Annotation split:    {args.annotation_split}")
    print(f"Random seed:         {args.seed}")
    print(f"Selected samples:    {len(loaded)}")
    for name in selected_names:
        print(f"  - {name}")
    print(f"Parameter settings:  {len(grid)}")
    print("Reference:           manual integer instance masks")
    print("Ranking:             mean instance F1@0.50, then mean semantic IoU")
    print("Model normalization: disabled")

    rows: list[dict[str, Any]] = []
    for parameter_index, parameters in enumerate(grid, start=1):
        print(f"\n[{parameter_index:03d}/{len(grid):03d}] {parameters}")
        parameter_rows: list[dict[str, Any]] = []
        for item, image, reference in loaded:
            prediction_cfg = PredictionConfig(**asdict(base_cfg))
            for key, value in parameters.items():
                setattr(prediction_cfg, key, value)
            prediction = predict_one(model, image, prediction_cfg)

            metrics: dict[str, Any] = {}
            metrics.update(_semantic_metrics(reference, prediction))
            metrics.update(_instance_metrics(reference, prediction))

            sample_dir = output_root / f"parameter_{parameter_index:03d}" / Path(item.sample_key)
            comparison_path = sample_dir / "comparison.png"
            row: dict[str, Any] = {
                "parameter_index": parameter_index,
                "sample_key": item.sample_key,
                **parameters,
                **metrics,
                "manual_mask": str(item.training_mask),
                "prepared_input": str(item.input_zarr),
                "comparison_png": str(comparison_path),
            }
            _save_comparison(
                comparison_path,
                image,
                reference,
                prediction,
                row,
                dataset=dataset,
                max_size=args.max_preview_size,
            )
            if args.save_labels:
                sample_dir.mkdir(parents=True, exist_ok=True)
                max_label = int(np.max(prediction)) if prediction.size else 0
                dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
                tiff.imwrite(
                    sample_dir / "predicted_labels.tif",
                    prediction.astype(dtype, copy=False),
                    photometric="minisblack",
                )
            parameter_rows.append(row)
            rows.append(row)
            print(
                f"  {item.sample_key}: semantic IoU={row['semantic_iou']:.4f}, "
                f"Dice={row['semantic_dice']:.4f}, "
                f"instance F1@0.50={row['instance_f1_50']:.4f}, "
                f"manual/predicted={row['reference_instances']}/{row['predicted_instances']}"
            )

        aggregate: dict[str, Any] = {
            "parameter_index": parameter_index,
            "sample_key": "__MEAN__",
            **parameters,
        }
        numeric_keys = [
            key
            for key, value in parameter_rows[0].items()
            if key
            not in {
                "parameter_index",
                "sample_key",
                "manual_mask",
                "prepared_input",
                "comparison_png",
                *parameters.keys(),
            }
            and isinstance(value, (int, float, np.integer, np.floating))
        ]
        for key in numeric_keys:
            aggregate[key] = _mean_numeric(parameter_rows, key)
        aggregate.update(
            {
                "manual_mask": "",
                "prepared_input": "",
                "comparison_png": "",
            }
        )
        rows.append(aggregate)

    aggregate_rows = [row for row in rows if row["sample_key"] == "__MEAN__"]
    best = max(
        aggregate_rows,
        key=lambda row: (row["instance_f1_50"], row["semantic_iou"], row["semantic_dice"]),
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
        "reference_type": "manual_integer_instance_masks",
        "annotation_source": args.annotation_source,
        "annotation_split": args.annotation_split,
        "random_selection": not bool(selected_keys),
        "sample_count": len(loaded),
        "seed": args.seed,
        "distinct_source_images": not args.allow_same_source,
        "selected_samples": selected_names,
        "invalid_pairs_skipped": invalid_pairs,
        "best_parameters_and_metrics": best,
        "ranking": "mean instance F1@0.50, then mean semantic IoU, then mean Dice",
        "model_normalization": False,
        "metrics_csv": str(csv_path),
    }
    best_path = output_root / "best_parameters.json"
    best_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    html_path = output_root / "screening_report.html"
    _write_html(html_path, rows, best)

    print("\nBest mean setting")
    print("=" * 78)
    print(json.dumps(best, indent=2, default=str))
    print(f"\nMetrics: {csv_path}")
    print(f"Report:  {html_path}")
    print(f"Summary: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
