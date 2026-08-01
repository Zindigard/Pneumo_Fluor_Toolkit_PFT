"""Validate manual full-image or crop annotations before fine-tuning.

The checker verifies that every selected image/mask pair exists, has matching
YX dimensions, uses finite float32 image data in [0, 1], and contains positive
integer instance labels. It also summarizes explicit train/validation crop
assignments and warns when both splits originate from the same source image.
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
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.segmentation.instance_segmentation_core import (  # noqa: E402
    ANNOTATION_SOURCES,
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
    normalized = key.replace("\\", "/")
    if "/crops/train/" in normalized:
        return "train"
    if "/crops/validation/" in normalized:
        return "validation"
    return "unspecified_full_image"


def _source_sample(key: str) -> str:
    return key.replace("\\", "/").split("/crops/", 1)[0]


def main() -> int:
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
    rows: list[dict[str, object]] = []
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
            if split == "unspecified_full_image":
                row["status"] = "WARN"
                row["message"] = (
                    "Full-image annotation has no explicit train/validation assignment."
                )
                warnings += 1
        except Exception as exc:
            row["status"] = "FAIL"
            row["message"] = str(exc)
            failures += 1
        rows.append(row)
        print(f"[{index:03d}/{len(items):03d}] {row['status']} {item.sample_key}")
        if row["message"]:
            print(f"  {row['message']}")

    train_sources = {
        str(row["source_sample"])
        for row in rows
        if row["status"] != "FAIL" and row["annotation_split"] == "train"
    }
    validation_sources = {
        str(row["source_sample"])
        for row in rows
        if row["status"] != "FAIL" and row["annotation_split"] == "validation"
    }
    overlap = sorted(train_sources & validation_sources)
    train_count = sum(row["annotation_split"] == "train" and row["status"] != "FAIL" for row in rows)
    validation_count = sum(
        row["annotation_split"] == "validation" and row["status"] != "FAIL"
        for row in rows
    )
    if validation_count == 0:
        warnings += 1
    if overlap:
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
        "pair_count": len(rows),
        "train_pair_count": train_count,
        "validation_pair_count": validation_count,
        "failure_count": failures,
        "warning_count": warnings,
        "train_source_samples": sorted(train_sources),
        "validation_source_samples": sorted(validation_sources),
        "source_sample_overlap": overlap,
        "messages": [
            "No explicit validation crops exist."
            if validation_count == 0
            else "Explicit validation crops found.",
            (
                "Training and validation use crops from the same source images: "
                + ", ".join(overlap)
            )
            if overlap
            else "No source-image overlap between explicit training and validation crops.",
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
                f"Training: {train_count}",
                f"Validation: {validation_count}",
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
    print(f"Pairs:       {len(rows)}")
    print(f"Training:    {train_count}")
    print(f"Validation:  {validation_count}")
    print(f"Failures:    {failures}")
    print(f"Warnings:    {warnings}")
    if overlap:
        print("WARNING: train/validation crops share source images:")
        for sample in overlap:
            print(f"  {sample}")
    print(f"Reports:     {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
