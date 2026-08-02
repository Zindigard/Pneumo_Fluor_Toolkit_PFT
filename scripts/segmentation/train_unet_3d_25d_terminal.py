"""
Train the per-volume-target merged-RGB 2.5D foreground U-Net.

Each annotated stack contributes one target slice from the shared mapping. For
configured target Zn, the input is merged-RGB Z(n-1)/Zn/Z(n+1), concatenated
as nine channels, and the target is ``zNNN_mask.tif``.

Examples
--------
Show all options:

    python scripts/segmentation/train_unet_3d_25d_terminal.py --help

Train the 2.5D U-Net for 50 epochs:

    python scripts/segmentation/train_unet_3d_25d_terminal.py \
        --level 0 \
        --patch 256 \
        --batch 4 \
        --epochs 50 \
        --steps-per-epoch 250 \
        --val-steps 50 \
        --val-split 0.2 \
        --learning-rate 0.001 \
        --predict-batch-size 8 \
        --seed 1337
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import (  # noqa: E402
    MODEL_CONTRACT,
    UNet25DTrainConfig,
    train_3d_25d_unet,
)


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Train the PFT 2.5D U-Net from one configured target slice per annotated volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "results" / "img" / "3d_data")
    parser.add_argument("--mask-root", type=Path, default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models" / "u_net_3d_25d")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    config = UNet25DTrainConfig(
        project_root=PROJECT_ROOT,
        image_root=args.image_root,
        mask_root=args.mask_root,
        model_root=args.model_root,
        level=args.level,
        patch=args.patch,
        batch=args.batch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        val_split=args.val_split,
        seed=args.seed,
        lr=args.learning_rate,
        base_filters=args.base_filters,
        dropout=args.dropout,
        channels=None,
        threshold=0.5,
        predict_batch_size=args.predict_batch_size,
        z_radius=1,
    )

    print("\nPFT per-volume-target 2.5D U-Net training")
    print("=" * 72)
    print(f"Model contract:      {MODEL_CONTRACT}")
    print("Target policy:       one configured target slice per annotated volume")
    print("Missing masks:       skipped with a warning; never treated as background")
    print("Training context:    Z-1/Z/Z+1 merged RGB -> target Z")
    print("Model input:         9 channels")
    print(f"Image root:          {config.image_root}")
    print(f"Mask root:           {config.mask_root}")
    print(f"Model root:          {config.model_root}")

    outputs = train_3d_25d_unet(config)
    print("\nTraining completed")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
