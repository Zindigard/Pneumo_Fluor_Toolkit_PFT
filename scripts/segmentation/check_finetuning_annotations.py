"""
Validate manual full-image or crop annotations before fine-tuning.

The checker verifies that every selected image/mask pair exists, has matching
YX dimensions, uses finite float32 image data in [0, 1], and contains positive
integer instance labels. It also previews the source-aware training and
validation split used by the fine-tuning scripts. Explicit crop assignments are
preserved, while the requested validation fraction is applied to unspecified
source images.

Examples
--------
Show all command-line parameters:

    python scripts/segmentation/check_finetuning_annotations.py --help

Representative execution:

    python scripts/segmentation/check_finetuning_annotations.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --annotation-source all \
        --validation-policy combined \
        --validation-fraction 0.2 \
        --seed 1337
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    VALIDATION_POLICIES,
    _split_train_validation,
    list_training_inputs,
    load_instance_mask,
    load_prepared_image,
)
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
    segmentation_mask_root,
)


def _split_from_key(key: str) -> str:
    """Split from key into the requested subsets.

    Args:
        key (str): Key used to access or identify an entry in a mapping.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _split_from_key(key="key")
    """
    normalized = key.replace("\\", "/")
    if "/crops/train/" in normalized:
        return "train"
    if "/crops/validation/" in normalized:
        return "validation"
    return "unspecified_full_image"


def _source_sample(key: str) -> str:
    """Return source sample for the supplied inputs.

    Args:
        key (str): Key used to access or identify an entry in a mapping.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _source_sample(key="key")
    """
    return key.replace("\\", "/").split("/crops/", 1)[0]


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Check manual image/mask pairs before Cellpose or Omnipose fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
        required=True,
    )
    parser.add_argument(
        "--annotation-source",
        choices=ANNOTATION_SOURCES,
        default="crops-only",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-policy",
        choices=VALIDATION_POLICIES,
        default="combined",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    valid_modes = SOURCE_MODES_BY_DATASET[args.dataset]
    if args.source_mode not in valid_modes:
        parser.error(f"{args.source_mode!r} is invalid for {args.dataset!r}: {valid_modes}")

    items = list_training_inputs(
        PROJECT_ROOT,
        args.dataset,
        args.source_mode,
        annotation_source=args.annotation_source,
    )
    crop_sources_with_masks = {
        _source_sample(item.sample_key)
        for item in items
        if "/crops/" in item.sample_key.replace("\\", "/")
        and item.training_mask.is_file()
    }
    rows: list[dict[str, object]] = []
    valid_images: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    valid_names: list[str] = []
    failures = 0
    warnings = 0
    for index, item in enumerate(items, start=1):
        split = _split_from_key(item.sample_key)
        row: dict[str, object] = {
            "sample_key": item.sample_key,
            "source_sample": _source_sample(item.sample_key),
            "annotation_split": split,
            "input_zarr": str(item.input_zarr),
            "mask_tif": str(item.training_mask),
            "status": "PASS",
            "message": "",
        }
        try:
            if not item.training_mask.is_file():
                if (
                    args.annotation_source == "all"
                    and split == "unspecified_full_image"
                    and _source_sample(item.sample_key) in crop_sources_with_masks
                ):
                    row["status"] = "SKIP"
                    row["message"] = (
                        "Complete-image mask is absent; valid crop annotations from "
                        "this source image will be used instead."
                    )
                    rows.append(row)
                    print(
                        f"[{index:03d}/{len(items):03d}] "
                        f"{row['status']} {item.sample_key}"
                    )
                    print(f"  {row['message']}")
                    continue
                raise FileNotFoundError(f"Mask is missing: {item.training_mask}")
            image = load_prepared_image(item.input_zarr)
            mask = load_instance_mask(item.training_mask, image.shape[:2])
            labels = np.unique(mask[mask > 0])
            row.update(
                {
                    "image_shape": str(tuple(image.shape)),
                    "image_dtype": str(image.dtype),
                    "image_min": float(np.min(image)),
                    "image_max": float(np.max(image)),
                    "mask_shape": str(tuple(mask.shape)),
                    "mask_dtype": str(mask.dtype),
                    "instance_count": int(labels.size),
                    "maximum_label": int(labels.max()) if labels.size else 0,
                }
            )
            valid_images.append(image)
            valid_masks.append(mask)
            valid_names.append(item.sample_key)
            # Unspecified complete images are valid. The combined policy assigns
            # them reproducibly after all pairs have been checked.
        except Exception as exc:
            row["status"] = "FAIL"
            row["message"] = str(exc)
            failures += 1
        rows.append(row)
        print(f"[{index:03d}/{len(items):03d}] {row['status']} {item.sample_key}")
        if row["message"]:
            print(f"  {row['message']}")

    split_diagnostics: dict[str, object] = {}
    split_error = ""
    planned_train_count = 0
    planned_validation_count = 0
    train_sources: set[str] = set()
    validation_sources: set[str] = set()
    overlap: list[str] = []
    if valid_names:
        try:
            _train, _validation, split_diagnostics = _split_train_validation(
                valid_images,
                valid_masks,
                valid_names,
                args.validation_fraction,
                args.seed,
                args.validation_policy,
            )
            planned_train_count = int(split_diagnostics["train_pair_count"])
            planned_validation_count = int(
                split_diagnostics["validation_pair_count"]
            )
            train_sources = set(split_diagnostics["train_source_samples"])
            validation_sources = set(
                split_diagnostics["validation_source_samples"]
            )
            overlap = list(split_diagnostics["source_sample_overlap"])
            split_by_name = {
                str(record["sample_key"]): record
                for record in split_diagnostics.get("split_rows", [])
            }
            for row in rows:
                record = split_by_name.get(str(row["sample_key"]))
                if record is not None:
                    row["planned_split"] = record["final_split"]
                    row["assignment_reason"] = record["assignment_reason"]
        except Exception as exc:
            split_error = str(exc)
            failures += 1

    if planned_validation_count == 0 and not split_error:
        warnings += 1

    output_dir = (
        segmentation_mask_root(PROJECT_ROOT, args.dataset, args.source_mode)
        / "checks"
        / "finetuning"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "finetuning_annotation_check.csv"
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "source_mode": args.source_mode,
        "annotation_source": args.annotation_source,
        "validation_fraction": args.validation_fraction,
        "validation_policy": args.validation_policy,
        "seed": args.seed,
        "pair_count": len(rows),
        "valid_pair_count": len(valid_names),
        "planned_train_pair_count": planned_train_count,
        "planned_validation_pair_count": planned_validation_count,
        "failure_count": failures,
        "warning_count": warnings,
        "train_source_samples": sorted(train_sources),
        "validation_source_samples": sorted(validation_sources),
        "source_sample_overlap": overlap,
        "split_error": split_error,
        "split_diagnostics": split_diagnostics,
        "messages": [
            (
                f"Planned source-aware split: {planned_train_count} training and "
                f"{planned_validation_count} validation pairs."
            )
            if not split_error
            else f"Split construction failed: {split_error}",
            "No source-image overlap between training and validation."
            if not overlap
            else "Source-image overlap detected: " + ", ".join(overlap),
        ],
        "rows": rows,
    }
    json_path = output_dir / "finetuning_annotation_check.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    text_path = output_dir / "finetuning_annotation_check.txt"
    text_path.write_text(
        "\n".join(
            [
                "PFT FINE-TUNING ANNOTATION CHECK",
                "=" * 72,
                f"Dataset: {args.dataset}",
                f"Source mode: {args.source_mode}",
                f"Annotation source: {args.annotation_source}",
                f"Pairs: {len(rows)}",
                f"Valid pairs: {len(valid_names)}",
                f"Planned training: {planned_train_count}",
                f"Planned validation: {planned_validation_count}",
                f"Failures: {failures}",
                f"Warnings: {warnings}",
                *summary["messages"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nSummary")
    print("=" * 72)
    print(f"Pairs:               {len(rows)}")
    print(f"Valid pairs:         {len(valid_names)}")
    print(f"Planned training:    {planned_train_count}")
    print(f"Planned validation:  {planned_validation_count}")
    print(f"Failures:            {failures}")
    print(f"Warnings:            {warnings}")
    if split_error:
        print(f"SPLIT ERROR: {split_error}")
    if overlap:
        print("WARNING: training and validation share source images:")
        for sample in overlap:
            print(f"  {sample}")
    print(f"Reports:     {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
