"""Fine-tune Cellpose-SAM, Omnipose, or StarDist on PFT annotations.

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
        validation/
            validation_metrics.csv
            validation_summary.json
            <sample>/comparison.png
            <sample>/predicted_labels.tif

The validation metrics are binary Dice and IoU. Instance F1 and split/merge
metrics are intentionally not calculated for this thesis workflow.
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
    ANNOTATION_SOURCES,
    MODEL_FAMILIES,
    TrainingConfig,
    collect_training_data,
    train_initial_model,
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


def _default_run_name(family: str, dataset: str, source_mode: str) -> str:
    return f"{family}_{dataset}_{source_mode}_finetuned_v1"


def main(default_family: str | None = None) -> int:
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
    parser.add_argument("--validation-min-size", type=int, default=15)
    parser.add_argument("--no-validation-labels", action="store_true")
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
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

    images, _masks, names = collect_training_data(
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
    print("Validation metrics:     binary Dice and IoU")
    print(f"Epochs / batch size:    {epochs} / {batch_size}")
    print(f"Learning rate / WD:     {learning_rate:g} / {weight_decay:g}")
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
        seed=args.seed,
        save_every=args.save_every,
        min_train_masks=args.min_train_masks,
        n_rays=args.n_rays,
        grid=args.grid,
        steps_per_epoch=args.steps_per_epoch,
        patch_size=tuple(args.patch_size),
        annotation_source=args.annotation_source,
        require_validation=not args.allow_no_validation,
        validation_diameter=args.validation_diameter,
        validation_flow_threshold=args.validation_flow_threshold,
        validation_cellprob_threshold=args.validation_cellprob_threshold,
        validation_mask_threshold=args.validation_mask_threshold,
        validation_min_size=args.validation_min_size,
        save_validation_labels=not args.no_validation_labels,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
