"""Tune prediction parameters against manual PFT instance masks.

The script evaluates a family-specific parameter grid and ranks settings by
mean instance F1 at IoU 0.5, followed by semantic Dice and IoU. It does not
retrain the network and it never renormalizes the prepared input.

Results are saved below::

    models/segmentation/<family>/<dataset>/<source_mode>/tuning/<model>/
        parameter_tuning.csv
        best_parameters.json

Examples
--------
Tune Cellpose::

    python scripts/segmentation/tune_segmentation_model.py `
        --family cellpose `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --model cpsam `
        --flow-thresholds 0.2 0.4 0.6 `
        --cellprob-thresholds -1 0 1

Tune StarDist::

    python scripts/segmentation/tune_segmentation_model.py `
        --family stardist `
        --dataset 3d_mip `
        --source-mode deconv_masked `
        --model "models/segmentation/stardist/3d_mip/deconv_masked/my_run" `
        --prob-thresholds 0.3 0.5 0.7 `
        --nms-thresholds 0.3 0.4 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

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
    MODEL_FAMILIES,
    TuneConfig,
    collect_training_data,
    tune_model_parameters,
)
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
)


def _choose(title: str, values: Sequence[str]) -> str:
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _diameters(values: list[str]) -> tuple[float | None, ...]:
    output: list[float | None] = []
    for value in values:
        output.append(None if value.lower() in {"none", "auto"} else float(value))
    return tuple(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune model inference parameters using manual instance masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family", choices=MODEL_FAMILIES)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5)
    parser.add_argument("--diameters", nargs="+", default=["none"])
    parser.add_argument("--flow-thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--cellprob-thresholds", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--mask-thresholds", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--prob-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--nms-thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    family = args.family or _choose("Choose model family", MODEL_FAMILIES)
    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is invalid for {dataset!r}: {valid_modes}")

    selected = set(args.sample) if args.sample else None
    _images, _masks, names = collect_training_data(
        PROJECT_ROOT,
        dataset,
        source_mode,
        selected_sample_keys=selected,
    )
    print("\nPFT instance-segmentation parameter tuning")
    print("=" * 72)
    print(f"Family:              {family}")
    print(f"Dataset:             {dataset}")
    print(f"Source mode:         {source_mode}")
    print(f"Validation masks:    {len(names)}")
    print("Ranking metric:      mean instance F1 at selected IoU threshold")
    print("Model normalization: DISABLED")

    cfg = TuneConfig(
        project_root=PROJECT_ROOT,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        model=args.model,
        model_name=args.model_name,
        gpu=not args.cpu,
        selected_sample_keys=selected,
        instance_iou_threshold=args.instance_iou_threshold,
        diameter_values=_diameters(args.diameters),
        flow_threshold_values=tuple(args.flow_thresholds),
        cellprob_threshold_values=tuple(args.cellprob_thresholds),
        mask_threshold_values=tuple(args.mask_thresholds),
        prob_thresh_values=tuple(args.prob_thresholds),
        nms_thresh_values=tuple(args.nms_thresholds),
        min_size=args.min_size,
        batch_size=args.batch_size,
    )
    output = tune_model_parameters(cfg)
    best_path = output / "best_parameters.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))["best_parameters_and_metrics"]
    print(f"\nBest parameters and mean metrics:\n{json.dumps(best, indent=2)}")
    print(f"\nSaved tuning outputs to:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
