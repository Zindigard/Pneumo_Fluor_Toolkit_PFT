"""
Interactive launcher for the validated PFT instance-segmentation workflow.

The launcher deliberately calls the documented terminal scripts as separate
processes. This keeps preparation, validation, labeling, prediction, training,
and parameter tuning reproducible and prevents the legacy runners from
normalizing sparse U-Net outputs again.

Examples
--------
Launch the interactive workflow menu:

    python scripts/segmentation/segmentation_launcher.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ACTIONS: tuple[tuple[str, str], ...] = (
    ("Prepare normalized segmentation inputs", "prepare_segmentation_inputs.py"),
    ("Check sources, metadata, dtype, and masks", "check_segmentation_inputs.py"),
    ("Create or edit manual instance masks in Napari", "labeling_segmentation.py"),
    ("Check full-image and crop annotations before fine-tuning", "check_finetuning_annotations.py"),
    ("Screen pretrained Cellpose-SAM or Omnipose against U-Net masks", "screen_pretrained_segmentation.py"),
    ("Run Cellpose, Omnipose, or StarDist prediction", "run_segmentation_model.py"),
    ("Train or fine-tune an initial model", "train_segmentation_model.py"),
    ("Tune prediction parameters on manual validation masks", "tune_segmentation_model.py"),
)


def _choose_action() -> tuple[str, str]:
    """Choose action according to the configured criteria.

    Returns:
        tuple[str, str]: Collection containing the generated or selected values.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _choose_action()
    """
    print("\nPFT instance-segmentation workflow")
    print("=" * 72)
    for index, (label, _script) in enumerate(ACTIONS, start=1):
        print(f"  [{index}] {label}")
    print("  [0] Exit")

    while True:
        value = input("Select number: ").strip()
        try:
            selected = int(value)
        except ValueError:
            print("Enter a number from the menu.")
            continue
        if selected == 0:
            raise SystemExit(0)
        if 1 <= selected <= len(ACTIONS):
            return ACTIONS[selected - 1]
        print(f"Selection must be between 0 and {len(ACTIONS)}.")


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    label, script_name = _choose_action()
    script_path = SCRIPT_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Workflow script was not found: {script_path}")

    print(f"\nStarting: {label}")
    completed = subprocess.run([sys.executable, str(script_path)], check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
