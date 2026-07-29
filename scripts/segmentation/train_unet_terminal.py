"""Train the thesis-aligned 2D U-Net from terminal prompts or argparse."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

_SCRIPT = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (_SCRIPT.parent, *_SCRIPT.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate project root containing scripts and src/PFT")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (  # noqa: E402
    DATASETS_2D,
    UNet2DTrainConfig,
    default_filtered_root,
    train_2d_time_unet,
)
from PFT.core_prog_parts.segmentation.unet_train_2d_wga_dapi_core import (  # noqa: E402
    UNet2DWgaDapiTrainConfig,
    train_2d_wga_dapi_unet,
)


def _choose_dataset() -> str:
    print("\nChoose 2D U-Net training dataset:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    answer = input("Choose number [1]: ").strip()
    return "2d_wga_dapi" if answer == "2" else "2d_time"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 2D bacterial foreground/background U-Net."
    )
    parser.add_argument("--dataset", choices=DATASETS_2D)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--val-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--foreground-patch-fraction",
        type=float,
        default=0.75,
        help="Requested fraction of positive/mask-containing training patches.",
    )
    parser.add_argument(
        "--foreground-min-ratio",
        type=float,
        default=0.01,
        help="Minimum labelled foreground fraction in a positive patch.",
    )
    parser.add_argument(
        "--background-max-ratio",
        type=float,
        default=0.001,
        help="Maximum labelled foreground fraction in a background patch.",
    )
    parser.add_argument(
        "--sampling-max-tries",
        type=int,
        default=160,
        help="Maximum crop attempts per requested class-aware patch.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=64,
        help="Training-patch shuffle buffer. A smaller value starts faster on CPU.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=6,
        help="Maximum normalized image-mask pairs retained in the in-memory LRU cache.",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=4,
        help="Epochs without validation-loss improvement before halving the learning rate.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Epochs without validation-loss improvement before stopping and restoring best weights.",
    )
    parser.add_argument(
        "--dynamic-validation",
        action="store_true",
        help="Resample validation patches continuously instead of reusing a fixed validation set.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require command-line choices and never open a prompt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset is None:
        if args.non_interactive:
            raise SystemExit("--dataset is required with --non-interactive")
        dataset = _choose_dataset()
    else:
        dataset = args.dataset

    config_class = (
        UNet2DWgaDapiTrainConfig if dataset == "2d_wga_dapi" else UNet2DTrainConfig
    )
    cfg = config_class(project_root=PROJECT_ROOT, dataset=dataset)
    cfg.image_root = Path(args.image_root or default_filtered_root(PROJECT_ROOT, dataset))
    cfg.mask_root = Path(
        args.mask_root
        or PROJECT_ROOT / "results" / "training_files" / "U-net" / dataset
    )
    cfg.model_root = Path(
        args.model_root or PROJECT_ROOT / "models" / f"u_net_{dataset}"
    )
    cfg.level = args.level
    cfg.patch = args.patch
    cfg.batch = args.batch
    cfg.epochs = args.epochs
    cfg.steps_per_epoch = args.steps_per_epoch
    cfg.val_steps = args.val_steps
    cfg.lr = args.learning_rate
    cfg.base_filters = args.base_filters
    cfg.dropout = args.dropout
    cfg.fg_fraction = args.foreground_patch_fraction
    cfg.fg_min_ratio = args.foreground_min_ratio
    cfg.bg_max_ratio = args.background_max_ratio
    cfg.max_tries = args.sampling_max_tries
    cfg.shuffle_buffer = args.shuffle_buffer
    cfg.cache_size = args.cache_size
    cfg.reduce_lr_patience = args.reduce_lr_patience
    cfg.early_stopping_patience = args.early_stopping_patience
    cfg.fixed_validation = not args.dynamic_validation
    cfg.seed = args.seed

    print("\n=== 2D U-NET TRAINING CONFIGURATION ===")
    print(f"Dataset:              {cfg.dataset}")
    print(f"Filtered input root:   {cfg.image_root}")
    print(f"Reference mask root:   {cfg.mask_root}")
    print(f"Model output root:     {cfg.model_root}")
    print("Stored input values:   original scale, not pre-normalized")
    print("Runtime normalization: complete image, per channel, P1-P99.8 -> [0,1]")
    print(f"Patch / batch:         {cfg.patch} / {cfg.batch}")
    print(f"Epochs:                {cfg.epochs}")
    print(f"Steps / val steps:     {cfg.steps_per_epoch} / {cfg.val_steps}")
    print(f"Learning rate:         {cfg.lr}")
    print(f"Foreground patches:    {cfg.fg_fraction:.1%}")
    print(f"Positive min ratio:    {cfg.fg_min_ratio:.4f}")
    print(f"Background max ratio:  {cfg.bg_max_ratio:.4f}")
    print(f"Sampling max tries:    {cfg.max_tries}")
    print(f"Shuffle buffer:        {cfg.shuffle_buffer}")
    print(f"Image cache size:      {cfg.cache_size}")
    print(f"Fixed validation:      {cfg.fixed_validation}")
    if cfg.fixed_validation:
        print(f"Fixed val patches:     {cfg.val_steps * cfg.batch}")
    print(f"Reduce-LR patience:    {cfg.reduce_lr_patience}")
    print(f"Early-stop patience:   {cfg.early_stopping_patience}")

    outputs = (
        train_2d_wga_dapi_unet(cfg)
        if dataset == "2d_wga_dapi"
        else train_2d_time_unet(cfg)
    )
    print("\nTraining finished. Saved outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
