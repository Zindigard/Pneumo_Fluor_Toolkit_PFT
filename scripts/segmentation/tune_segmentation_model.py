r"""Tune inference parameters after fine-tuning using manual validation masks.

Only binary semantic Dice and IoU are calculated. Instance F1 and split/merge
metrics are not part of this thesis workflow.

By default, the script evaluates only crop annotations stored under
``crops/validation``. Each parameter/sample pair receives a comparison PNG and
predicted-label TIFF.

Examples
--------
Show all command-line parameters:

    python scripts/segmentation/tune_segmentation_model.py --help

Representative execution:

    python scripts/segmentation/tune_segmentation_model.py \
        --dataset 2d_time \
        --family omnipose \
        --source-mode filtered_unet \
        --annotation-source all \
        --annotation-split validation \
        --sample WT_HADA_NHS_40min_ROI1_SIM \
        --model models/example_model \
        --model-name example_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    """Return project root for the supplied inputs.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _project_root()
    """
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
    """Choose the requested operation according to the configured criteria.

    Args:
        title (str): Title displayed on the generated figure or report section.
        values (Sequence[str]): Text value specifying values.

    Returns:
        str: Generated or resolved text value.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _choose(title="title", values="values")
    """
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _diameters(values: list[str]) -> tuple[float | None, ...]:
    """Return diameters for the supplied inputs.

    Args:
        values (list[str]): Text value specifying values.

    Returns:
        tuple[float | None, ...]: Collection containing the generated or selected values.

    Example:
        >>> result = _diameters(values="values")
    """
    return tuple(
        None if value.lower() in {"none", "auto"} else float(value)
        for value in values
    )


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Tune model inference parameters using manual validation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family", choices=MODEL_FAMILIES)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--model", required=False, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument(
        "--annotation-source",
        choices=ANNOTATION_SOURCES,
        default="crops-only",
    )
    parser.add_argument(
        "--annotation-split",
        choices=ANNOTATION_SPLITS,
        default="validation",
        help="Use explicit validation crops by default.",
    )
    parser.add_argument("--diameters", nargs="+", default=["none"])
    parser.add_argument("--flow-thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--cellprob-thresholds", type=float, nargs="+", default=[-1.0, -0.5, 0.0])
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
        annotation_source=args.annotation_source,
        annotation_split=args.annotation_split,
    )
    print("\nPFT fine-tuned-model parameter tuning")
    print("=" * 72)
    print(f"Family:              {family}")
    print(f"Dataset:             {dataset}")
    print(f"Source mode:         {source_mode}")
    print(f"Annotation source:   {args.annotation_source}")
    print(f"Annotation split:    {args.annotation_split}")
    print(f"Validation pairs:    {len(names)}")
    print("Ranking metric:      mean semantic IoU, then mean Dice")
    print("Instance F1:         not calculated")
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
        annotation_source=args.annotation_source,
        annotation_split=args.annotation_split,
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
    print(f"\nBest parameters and mean binary metrics:\n{json.dumps(best, indent=2)}")
    print(f"\nSaved tuning outputs to:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
