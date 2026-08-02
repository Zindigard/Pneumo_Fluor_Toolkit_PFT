"""Terminal entry point for training one two-dimensional Noise2Void model.

Run without ``--model`` to select the model interactively.  The command reads
only original OME-Zarr images from ``results/img`` and creates an image-level
train/validation split before patch extraction.

Examples
--------
Interactive model selection with default parameters::

    python scripts/denoising/train_n2v_2d.py

Train the 2D time-lapse model::

    python scripts/denoising/train_n2v_2d.py --model time

Train the DAPI model for 40 epochs with a smaller batch::

    python scripts/denoising/train_n2v_2d.py --model dapi --epochs 40 --batch-size 64

Replace an existing model directory intentionally::

    python scripts/denoising/train_n2v_2d.py --model wga --overwrite-model

Use ``--help`` to see every parameter and its effect on training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_FILE = Path(__file__).resolve()


def find_project_root() -> Path:
    """Locate the repository root containing both ``scripts`` and ``src/PFT``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_project_root()
    """

    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT repository root from this script.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.denoising.n2v_train_models import (  # noqa: E402
    N2VTrainingParameters,
    train_model,
)
from PFT.core_prog_parts.denoising.n2v_workflow import (  # noqa: E402
    MODEL_SPECS,
    choose_model_key_interactive,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser with scientifically explicit help text.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train one 2D Noise2Void model from original OME-Zarr images under results/img. "
            "Training and validation are split by complete image before patch extraction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        help="Model to train: time, DAPI, WGA, or joint DAPI+WGA.",
    )
    parser.add_argument("--patch-y", type=int, default=64, help="Patch height. Larger values use more context and GPU memory.")
    parser.add_argument("--patch-x", type=int, default=64, help="Patch width. Larger values use more context and GPU memory.")
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=8192,
        help="Requested patch sampling density. More patches increase coverage and preparation cost.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Fraction of complete images reserved for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the image split and patch previews.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs. Inspect validation loss for overfitting.")
    parser.add_argument("--batch-size", type=int, default=128, help="Patches per optimization step. Larger values require more GPU memory.")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optimizer steps per epoch. Omit to use ceil(training patches / batch size).",
    )
    parser.add_argument("--unet-kernel-size", type=int, default=3, help="Convolution kernel size.")
    parser.add_argument("--unet-first-features", type=int, default=64, help="Feature maps in the first U-Net level.")
    parser.add_argument("--unet-depth", type=int, default=3, help="Number of U-Net down/up-sampling levels.")
    parser.add_argument("--n2v-percent-pixels", type=float, default=0.198, help="Percentage of blind-spot target pixels per patch.")
    parser.add_argument("--n2v-neighborhood-radius", type=int, default=5, help="Radius for replacement sampling around blind-spot pixels.")
    parser.add_argument("--augment", action="store_true", help="Apply N2V geometric patch augmentation.")
    parser.add_argument("--shuffle-patches", action="store_true", help="Shuffle generated patches. The image split remains fixed.")
    parser.add_argument("--no-batch-norm", action="store_true", help="Disable batch normalization in the U-Net.")
    parser.add_argument(
        "--single-net-per-channel",
        action="store_true",
        help="For a joint input, request separate internal networks per channel.",
    )
    parser.add_argument(
        "--overwrite-model",
        action="store_true",
        help="Delete and replace an existing model folder. Use only for intentional retraining.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, optionally ask for a model, and execute one training run.

    Args:
        argv (list[str] | None): Optional command-line argument sequence. When omitted, arguments are read from ``sys.argv``. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """

    args = build_parser().parse_args(argv)
    model_key = args.model or choose_model_key_interactive("Choose the model to train")
    parameters = N2VTrainingParameters(
        patch_shape=(args.patch_y, args.patch_x),
        num_patches_per_image=args.patches_per_image,
        augment=args.augment,
        shuffle_patches=args.shuffle_patches,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        unet_kern_size=args.unet_kernel_size,
        unet_n_first=args.unet_first_features,
        unet_n_depth=args.unet_depth,
        batch_norm=not args.no_batch_norm,
        n2v_perc_pix=args.n2v_percent_pixels,
        n2v_neighborhood_radius=args.n2v_neighborhood_radius,
        single_net_per_channel=args.single_net_per_channel,
    )
    result = train_model(
        model_key,
        parameters=parameters,
        repo_root=PROJECT_ROOT,
        overwrite_model=args.overwrite_model,
    )
    print("\nTraining completed.")
    print(f"Model: {result.model_name}")
    print(f"Training curves: {result.artifact_directory / 'training_curves.png'}")
    print(f"Training history: {result.artifact_directory / 'training_history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
