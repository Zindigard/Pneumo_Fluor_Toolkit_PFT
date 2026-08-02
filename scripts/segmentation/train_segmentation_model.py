"""
Fine-tune Cellpose-SAM, Omnipose, or StarDist on PFT annotations.

The script supports complete-image annotations and crop-based annotations.
Crop annotations can be assigned explicitly to ``train`` or ``validation``
when they are created in Napari.

Recommended safety setting
--------------------------
For partially annotated large images, use ``--annotation-source crops-only``.
This prevents a partially labelled full-size image from entering training.

Outputs
-------
Models and reports are stored below::

    models/segmentation/<family>/<dataset>/<source_mode>/<run_name>/
        training_summary.json
        training_losses.csv                 # when returned by the library
        training_loss_curve.png             # when losses are available
        split_manifest.csv
        validation/
            baseline/validation_metrics.csv
            validation_metrics.csv
            validation_improvement_metrics.csv
            iou_improvement_by_sample.png
            iou_delta_by_sample.png
            mean_dice_iou_improvement.png
            validation_summary.json
            <sample>/comparison.png
            <sample>/predicted_labels.tif

The validation metrics are binary Dice and IoU. Instance F1 and split/merge
metrics are intentionally not calculated for this thesis workflow.

Examples
--------
Show all command-line parameters:

    python scripts/segmentation/train_segmentation_model.py --help

Representative execution:

    python scripts/segmentation/train_segmentation_model.py \
        --dataset 2d_time \
        --family omnipose \
        --source-mode filtered_unet \
        --annotation-source all \
        --run-name example_run \
        --pretrained-model bact_fluor_omni \
        --epochs 50 \
        --batch-size 8
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
    MODEL_FAMILIES,
    VALIDATION_POLICIES,
    TrainingConfig,
    _split_train_validation,
    collect_training_data,
    train_initial_model,
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


def _default_run_name(family: str, dataset: str, source_mode: str) -> str:
    """Return default run name for the supplied inputs.

    Args:
        family (str): Instance-segmentation model family to use, such as Cellpose, Omnipose, or StarDist.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _default_run_name(
        ...     family="cellpose",
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
    return f"{family}_{dataset}_{source_mode}_finetuned_v1"


def main(default_family: str | None = None) -> int:
    """Execute the command-line workflow and return its process exit status.

    Args:
        default_family (str | None): Text value specifying default family. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained instance-segmentation model.",
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
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--pretrained-model",
        default=None,
        help="Built-in name or explicit model path. Defaults to cpsam_v2/cpsam for Cellpose and bact_fluor_omni for Omnipose.",
    )
    parser.add_argument(
        "--annotation-source",
        choices=ANNOTATION_SOURCES,
        default="crops-only",
        help="Select crop annotations, complete-image annotations, or both.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-policy",
        choices=VALIDATION_POLICIES,
        default="combined",
        help=(
            "combined preserves explicit crop assignments and additionally holds "
            "out the requested fraction of unspecified source images"
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--min-train-masks", type=int, default=1)
    parser.add_argument(
        "--allow-no-validation",
        action="store_true",
        help="Allow training without a validation set. Automatic Dice/IoU validation will be skipped.",
    )
    parser.add_argument("--validation-diameter", type=float, default=None)
    parser.add_argument("--validation-flow-threshold", type=float, default=0.6)
    parser.add_argument("--validation-cellprob-threshold", type=float, default=-0.5)
    parser.add_argument("--validation-mask-threshold", type=float, default=0.0)
    parser.add_argument("--validation-prob-thresh", type=float, default=None)
    parser.add_argument("--validation-nms-thresh", type=float, default=None)
    parser.add_argument("--validation-min-size", type=int, default=15)
    parser.add_argument("--no-validation-labels", action="store_true")
    parser.add_argument(
        "--no-pretrained-baseline",
        action="store_true",
        help="Skip evaluation of the initial pretrained model on the validation set.",
    )
    parser.add_argument(
        "--no-improvement-graphs",
        action="store_true",
        help="Skip paired pretrained-versus-fine-tuned Dice/IoU graphs.",
    )
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument(
        "--foreground-patches-per-image",
        type=int,
        default=8,
        help="Maximum number of spatially distributed, mask-containing patches generated from each training annotation.",
    )
    parser.add_argument(
        "--stardist-foreground-fraction",
        type=float,
        default=1.0,
        help="Fraction of StarDist batches sampled from foreground-containing locations.",
    )
    args, extra_cli = parser.parse_known_args()

    family = args.family or _choose("Choose model family", MODEL_FAMILIES)
    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is invalid for {dataset!r}: {valid_modes}")
    run_name = args.run_name or _default_run_name(family, dataset, source_mode)
    family_defaults = {
        "cellpose": {"epochs": 100, "batch_size": 1, "learning_rate": 1e-5, "weight_decay": 0.1},
        "omnipose": {"epochs": 500, "batch_size": 8, "learning_rate": 0.2, "weight_decay": 1e-5},
        "stardist": {"epochs": 200, "batch_size": 4, "learning_rate": 3e-4, "weight_decay": 0.0},
    }[family]
    epochs = args.epochs if args.epochs is not None else family_defaults["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else family_defaults["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else family_defaults["learning_rate"]
    weight_decay = args.weight_decay if args.weight_decay is not None else family_defaults["weight_decay"]

    images, masks, names = collect_training_data(
        PROJECT_ROOT,
        dataset,
        source_mode,
        annotation_source=args.annotation_source,
    )
    print("\nPFT pretrained-model fine-tuning")
    print("=" * 72)
    print(f"Family:                 {family}")
    print(f"Dataset:                {dataset}")
    print(f"Source mode:            {source_mode}")
    print(f"Annotation source:      {args.annotation_source}")
    print(f"Available labelled pairs: {len(names)}")
    print(f"Stored input channels:  {1 if images[0].ndim == 2 else images[0].shape[-1]}")
    if family == "omnipose":
        print("Omnipose channels:      2d_time=HADA+zero; 2d_wga_dapi=WGA+DAPI")
        print("Omnipose architecture:  nchan=2, nclasses=3 for bact_fluor_omni")
    print("Input normalization:    already prepared; model normalization disabled")
    split_preview = _split_train_validation(
        images,
        masks,
        names,
        args.validation_fraction,
        args.seed,
        args.validation_policy,
    )[2]
    print("Validation metrics:     binary Dice and IoU")
    print(f"Validation policy:      {args.validation_policy}")
    print(
        "Planned split:          "
        f"{split_preview['train_pair_count']} train / "
        f"{split_preview['validation_pair_count']} validation pairs"
    )
    print(
        "Validation composition: "
        f"{len(split_preview['explicit_validation_sources'])} explicit source(s) + "
        f"{len(split_preview['fractional_validation_sources'])} fractional source(s)"
    )
    print(f"Epochs / batch size:    {epochs} / {batch_size}")
    print(f"Learning rate / WD:     {learning_rate:g} / {weight_decay:g}")
    if family in {"omnipose", "stardist"}:
        print(f"Mask-containing patches: enabled, max {args.foreground_patches_per_image} per training image")
    if family == "stardist":
        print(f"StarDist foreground sampling fraction: {args.stardist_foreground_fraction:g}")
    print(f"Run name:               {run_name}")

    cfg = TrainingConfig(
        project_root=PROJECT_ROOT,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        run_name=run_name,
        pretrained_model=args.pretrained_model,
        gpu=not args.cpu,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_fraction=args.validation_fraction,
        validation_policy=args.validation_policy,
        seed=args.seed,
        save_every=args.save_every,
        min_train_masks=args.min_train_masks,
        n_rays=args.n_rays,
        grid=args.grid,
        steps_per_epoch=args.steps_per_epoch,
        patch_size=tuple(args.patch_size),
        foreground_patches_per_image=args.foreground_patches_per_image,
        stardist_foreground_fraction=args.stardist_foreground_fraction,
        annotation_source=args.annotation_source,
        require_validation=not args.allow_no_validation,
        validation_diameter=args.validation_diameter,
        validation_flow_threshold=args.validation_flow_threshold,
        validation_cellprob_threshold=args.validation_cellprob_threshold,
        validation_mask_threshold=args.validation_mask_threshold,
        validation_prob_thresh=args.validation_prob_thresh,
        validation_nms_thresh=args.validation_nms_thresh,
        validation_min_size=args.validation_min_size,
        save_validation_labels=not args.no_validation_labels,
        evaluate_pretrained_baseline=not args.no_pretrained_baseline,
        save_improvement_graphs=not args.no_improvement_graphs,
        extra_cli=extra_cli,
    )
    output = train_initial_model(cfg)
    print(f"\nSaved fine-tuned model and reports to:\n{output}")
    summary_path = output / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        details = summary.get("details", {})
        split = details.get("split_diagnostics", {})
        if split.get("source_sample_overlap_warning"):
            print("\nWARNING: training and validation contain crops from the same source image:")
            for sample in split.get("source_sample_overlap", []):
                print(f"  {sample}")
            print("Prefer validation crops from separate original images when possible.")
        validation = details.get("automatic_validation", {})
        if validation.get("status") == "completed":
            print("\nAutomatic validation:")
            print(f"  mean Dice: {validation.get('mean_semantic_dice', float('nan')):.4f}")
            print(f"  mean IoU:  {validation.get('mean_semantic_iou', float('nan')):.4f}")
            print(f"  report:    {validation.get('metrics_csv')}")
        improvement = details.get("validation_improvement", {})
        if improvement.get("status") == "completed":
            print("\nPretrained-to-fine-tuned improvement:")
            print(
                f"  mean IoU:  {improvement.get('baseline_mean_semantic_iou', float('nan')):.4f} "
                f"-> {improvement.get('fine_tuned_mean_semantic_iou', float('nan')):.4f} "
                f"(delta {improvement.get('mean_iou_improvement', float('nan')):+.4f})"
            )
            print(
                f"  mean Dice: {improvement.get('baseline_mean_semantic_dice', float('nan')):.4f} "
                f"-> {improvement.get('fine_tuned_mean_semantic_dice', float('nan')):.4f} "
                f"(delta {improvement.get('mean_dice_improvement', float('nan')):+.4f})"
            )
            print(f"  IoU graph: {improvement.get('iou_improvement_png')}")
            print(f"  delta graph: {improvement.get('iou_delta_png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
