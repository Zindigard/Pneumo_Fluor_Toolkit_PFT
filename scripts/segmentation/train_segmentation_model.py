"""
Train or fine-tune an initial instance-segmentation model for one PFT dataset.

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/mask.tif

Fine-tune Cellpose for 2d_time::

    python scripts/segmentation/train_segmentation_model.py `
        --family cellpose `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --run-name cellpose_2d_time_v1 `
        --pretrained-model cpsam `
        --epochs 200

Train StarDist for deconvolved MIPs::

    python scripts/segmentation/train_segmentation_model.py `
        --family stardist `
        --dataset 3d_mip `
        --source-mode deconv_masked `
        --run-name stardist_deconv_mip_v1
"""

from __future__ import annotations

import argparse
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
    return f"{family}_{dataset}_{source_mode}_v1"


def main(default_family: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune an instance-segmentation model.",
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
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--min-train-masks", type=int, default=1)
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

    images, masks, names = collect_training_data(PROJECT_ROOT, dataset, source_mode)
    print("\nPFT initial instance-model training")
    print("=" * 72)
    print(f"Family:              {family}")
    print(f"Dataset:             {dataset}")
    print(f"Source mode:         {source_mode}")
    print(f"Training pairs:      {len(names)}")
    print(f"Input channels:      {1 if images[0].ndim == 2 else images[0].shape[-1]}")
    print("Input normalization: already prepared; model normalization disabled")
    print(f"Run name:            {run_name}")

    cfg = TrainingConfig(
        project_root=PROJECT_ROOT,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        run_name=run_name,
        pretrained_model=args.pretrained_model,
        gpu=not args.cpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        save_every=args.save_every,
        min_train_masks=args.min_train_masks,
        n_rays=args.n_rays,
        grid=args.grid,
        steps_per_epoch=args.steps_per_epoch,
        patch_size=tuple(args.patch_size),
        extra_cli=extra_cli,
    )
    output = train_initial_model(cfg)
    print(f"\nSaved model and training report to:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
