"""Train and evaluate the sparse Z10-only 2.5D foreground U-Net.

Each annotated target uses the wavelength-mapped merged RGB context
Z9/Z10/Z11 and the manual target mask at Z10. Each context slice is converted
to RGB using 561 nm=red, 488 nm=green, and 405 nm=blue. The three RGB images
are concatenated, giving nine model input channels. Only Z10 has a manual
binary target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.validation_3d import DEFAULT_TRAINING_SLICES_1BASED  # noqa: E402
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import (  # noqa: E402
    UNet25DTrainConfig,
    train_3d_25d_unet,
)


def _parse_slices(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.replace(",", " ").split()}))
    if values != DEFAULT_TRAINING_SLICES_1BASED:
        raise ValueError(
            f"This workflow is fixed to target slices {DEFAULT_TRAINING_SLICES_1BASED}; received {values}."
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the PFT merged-RGB 2.5D U-Net from Z10 annotations only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "results" / "img" / "3d_data")
    parser.add_argument("--mask-root", type=Path, default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models" / "u_net_3d_25d")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--slices", default=",".join(map(str, DEFAULT_TRAINING_SLICES_1BASED)))
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
        training_slices_1based=_parse_slices(args.slices),
        channels=None,
        patch=args.patch,
        batch=args.batch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        val_split=args.val_split,
        lr=args.learning_rate,
        base_filters=args.base_filters,
        dropout=args.dropout,
        predict_batch_size=args.predict_batch_size,
        threshold=0.5,
        z_radius=1,
        seed=args.seed,
    )
    print("\nPFT merged-RGB 2.5D U-Net training")
    print("=" * 72)
    print(f"Image root:          {config.image_root}")
    print(f"Mask root:           {config.mask_root}")
    print(f"Target slices:       {config.training_slices_1based}")
    print("Training context:    Z9/Z10/Z11 -> target Z10")
    print("Input representation: merged RGB at each context Z")
    print("Colour mapping:      561 nm=red, 488 nm=green, 405 nm=blue")
    print("Model input:         3 RGB images = 9 channels")
    print("Mask threshold:      0.5")

    outputs = train_3d_25d_unet(config)
    print("\nCompleted")
    for name, path in outputs.items():
        print(f"{name:20}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
