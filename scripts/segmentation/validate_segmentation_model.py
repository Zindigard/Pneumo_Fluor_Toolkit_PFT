"""
Validate a saved Omnipose or StarDist model without repeating training.

The validation split is reconstructed from the same annotation source,
validation policy, validation fraction, and random seed used during training.
Outputs include per-sample binary Dice and IoU, comparison PNG files, predicted
label TIFF files, and a standalone JSON summary.

Examples
--------
Show all command-line parameters:

    python scripts/segmentation/validate_segmentation_model.py --help

Representative execution:

    python scripts/segmentation/validate_segmentation_model.py \
        --model models/example_model \
        --dataset 2d_time \
        --family omnipose \
        --source-mode filtered_unet \
        --annotation-source all \
        --validation-policy combined \
        --validation-fraction 0.2 \
        --seed 1337
"""

from __future__ import annotations

import argparse
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
    MODEL_FAMILIES,
    VALIDATION_POLICIES,
    TrainingConfig,
    validate_saved_model,
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


def main(default_family: str | None = None) -> int:
    """Execute the command-line workflow and return its process exit status.

    Args:
        default_family (str | None): Text value specifying default family. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Validate one saved instance-segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family",
        choices=(default_family,) if default_family else MODEL_FAMILIES,
        default=default_family,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--annotation-source",
        choices=ANNOTATION_SOURCES,
        default="all",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-policy",
        choices=VALIDATION_POLICIES,
        default="combined",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--validation-diameter", type=float, default=None)
    parser.add_argument("--validation-flow-threshold", type=float, default=0.4)
    parser.add_argument("--validation-cellprob-threshold", type=float, default=-1.0)
    parser.add_argument("--validation-mask-threshold", type=float, default=0.0)
    parser.add_argument("--validation-prob-thresh", type=float, default=None)
    parser.add_argument("--validation-nms-thresh", type=float, default=None)
    parser.add_argument("--validation-min-size", type=int, default=15)
    parser.add_argument("--no-validation-labels", action="store_true")
    args = parser.parse_args()

    family = args.family or _choose("Choose model family", MODEL_FAMILIES)
    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is invalid for {dataset!r}: {valid_modes}")

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        parser.error(f"Saved model does not exist: {model_path}")

    cfg = TrainingConfig(
        project_root=PROJECT_ROOT,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        run_name=model_path.stem,
        gpu=not args.cpu,
        validation_fraction=args.validation_fraction,
        validation_policy=args.validation_policy,
        seed=args.seed,
        annotation_source=args.annotation_source,
        validation_diameter=args.validation_diameter,
        validation_flow_threshold=args.validation_flow_threshold,
        validation_cellprob_threshold=args.validation_cellprob_threshold,
        validation_mask_threshold=args.validation_mask_threshold,
        validation_prob_thresh=args.validation_prob_thresh,
        validation_nms_thresh=args.validation_nms_thresh,
        validation_min_size=args.validation_min_size,
        save_validation_labels=not args.no_validation_labels,
        evaluate_pretrained_baseline=False,
        save_improvement_graphs=False,
    )

    print("\nPFT saved-model validation")
    print("=" * 72)
    print(f"Family:             {family}")
    print(f"Dataset:            {dataset}")
    print(f"Source mode:        {source_mode}")
    print(f"Annotation source:  {args.annotation_source}")
    print(f"Validation policy:  {args.validation_policy}")
    print(f"Validation fraction:{args.validation_fraction}")
    print(f"Seed:               {args.seed}")
    print(f"Model:              {model_path}")

    result = validate_saved_model(cfg, model_path, args.output_dir)
    if result.get("status") != "completed":
        raise RuntimeError(f"Validation did not complete: {result}")

    print("\nValidation completed")
    print(f"Mean Dice: {result['mean_semantic_dice']:.4f}")
    print(f"Mean IoU:  {result['mean_semantic_iou']:.4f}")
    print(f"Report:    {result['metrics_csv']}")
    print(f"Output:    {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
