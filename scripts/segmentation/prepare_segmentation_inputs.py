"""
Create normalized OME-Zarr inputs for downstream instance segmentation.

    python scripts/segmentation/prepare_segmentation_inputs.py

All 2d_time images::

    python scripts/segmentation/prepare_segmentation_inputs.py `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --mode all

One deconvolved MIP::

    python scripts/segmentation/prepare_segmentation_inputs.py `
        --dataset 3d_mip `
        --source-mode deconv_masked `
        --mode one
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
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

from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
    discover_segmentation_sources,
    prepare_dataset_segmentation_inputs,
)


def _choose(title: str, values: Sequence[str]) -> str:
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    answer = input("Select number: ").strip()
    selected = int(answer)
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _write_summary(output_rows: list[dict[str, object]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "prepare_segmentation_inputs_summary.csv"
    if not output_rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(output_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare validated float32 inputs for Cellpose, Omnipose, or StarDist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--mode", choices=("one", "all"), default=None)
    parser.add_argument(
        "--sample",
        default=None,
        help="Exact sample key for --mode one; omitted opens a numbered selector",
    )
    parser.add_argument("--filter-root", type=Path, default=None)
    parser.add_argument("--unet-root", type=Path, default=None)
    parser.add_argument("--mip-root", type=Path, default=None)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.8)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is not valid for {dataset!r}: {valid_modes}")

    records = discover_segmentation_sources(
        PROJECT_ROOT,
        dataset,
        source_mode,
        filter_root=args.filter_root,
        unet_root=args.unet_root,
        mip_root=args.mip_root,
    )
    run_mode = args.mode or _choose("Process one sample or all samples", ("one", "all"))
    selected: set[str] | None = None
    if run_mode == "one":
        keys = [record.sample_key for record in records]
        sample = args.sample or _choose("Choose sample", keys)
        if sample not in keys:
            parser.error(f"Unknown sample key: {sample}")
        selected = {sample}

    print("\nSegmentation input preparation")
    print("=" * 72)
    print(f"Dataset:             {dataset}")
    print(f"Source mode:         {source_mode}")
    print(f"Samples selected:    {len(records) if selected is None else 1}")
    print(f"Normalization:       P{args.p_low:g}-P{args.p_high:g}, per numerical channel")
    print("Mask order:          normalize complete source, then apply mask")
    print("Saved dtype/range:   float32, [0, 1]")
    print("Model normalization: must remain disabled")

    outputs = prepare_dataset_segmentation_inputs(
        PROJECT_ROOT,
        dataset,
        source_mode,
        filter_root=args.filter_root,
        unet_root=args.unet_root,
        mip_root=args.mip_root,
        overwrite=not args.no_overwrite,
        p_low=args.p_low,
        p_high=args.p_high,
        selected_sample_keys=selected,
    )

    rows: list[dict[str, object]] = []
    for output in outputs:
        row = asdict(output)
        row = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in row.items()
        }
        rows.append(row)
        print(f"\nPASS {output.sample_key}")
        print(f"  output:   {output.output_zarr}")
        print(f"  shape:    {output.output_shape}")
        print(f"  range:    {output.output_min:.6g} .. {output.output_max:.6g}")
        print(f"  nonzero:  {output.nonzero_pixels}")

    summary = _write_summary(rows, outputs[0].output_zarr.parents[len(Path(outputs[0].sample_key).parts)])
    print(f"\nPrepared {len(outputs)} segmentation input(s).")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
