"""Create normalized OME-Zarr inputs for downstream instance segmentation.

The script keeps each biological dataset and MIP source mode independent.

Supported inputs
----------------
``2d_time`` and ``2d_wga_dapi``
    Uses the intensity-preserving local-threshold image together with the
    matching U-Net ``pred_mask.ome.zarr``. For ``2d_time``, numerical channel 0
    is always selected and any additional stored channel is ignored. The complete
    selected image is normalized first and the U-Net mask is applied afterwards.

``3d_mip``
    Uses one of the already created 2D MIP products:

    * ``raw_unmasked``
    * ``raw_masked``
    * ``deconv_masked``

For masked MIPs, the script reconstructs the unmasked intensity projection from
recorded provenance, normalizes that complete MIP, and only then reapplies the
recorded 2.5D U-Net mask. It never renormalizes the sparse masked MIP directly.

Outputs
-------
Each sample is saved as float32 in [0, 1] below::

    results/segmentation_inputs/<dataset>/<source_mode>/<sample>/
        segmentation_input.ome.zarr
        preview_raw.png
        preview_normalized.png
        preview_comparison.png

Examples
--------
Interactive::

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

from PFT.core_prog_parts.common_paths import resolve_project_path  # noqa: E402
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
    discover_segmentation_sources,
    prepare_dataset_segmentation_inputs,
)


def _choose(title: str, values: Sequence[str]) -> str:
    """Choose the requested operation according to the configured criteria.

    Args:
        title (str): Title displayed on the generated figure or report section.
        values (Sequence[str]): Text value specifying values.

    Returns:
        str: Generated or resolved text value.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _choose(title="title", values="values")
    """
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    answer = input("Select number: ").strip()
    selected = int(answer)
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _write_summary(output_rows: list[dict[str, object]], output_root: Path) -> Path:
    """Write summary to persistent storage.

    Args:
        output_rows (list[dict[str, object]]): Text value specifying output rows.
        output_root (Path): Directory used for output.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _write_summary(output_rows="output_rows", output_root=Path("path/to/resource"))
    """
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
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
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

    filter_root = (
        resolve_project_path(args.filter_root, PROJECT_ROOT)
        if args.filter_root is not None
        else None
    )
    unet_root = (
        resolve_project_path(args.unet_root, PROJECT_ROOT)
        if args.unet_root is not None
        else None
    )
    mip_root = (
        resolve_project_path(args.mip_root, PROJECT_ROOT)
        if args.mip_root is not None
        else None
    )

    records = discover_segmentation_sources(
        PROJECT_ROOT,
        dataset,
        source_mode,
        filter_root=filter_root,
        unet_root=unet_root,
        mip_root=mip_root,
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
    if dataset == "2d_time":
        print("Channel policy:       use C=0 only; ignore all additional channels")
    else:
        print("Channel policy:       retain the dataset-defined numerical channels")
    print("Mask order:          normalize complete source, then apply mask")
    print("Saved dtype/range:   float32, [0, 1]")
    print("Preview PNGs:       raw, normalized, and side-by-side comparison")
    print("Model normalization: must remain disabled")

    outputs = prepare_dataset_segmentation_inputs(
        PROJECT_ROOT,
        dataset,
        source_mode,
        filter_root=filter_root,
        unet_root=unet_root,
        mip_root=mip_root,
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
        print(f"  preview:  {output.preview_comparison_png}")

    summary = _write_summary(rows, outputs[0].output_zarr.parents[len(Path(outputs[0].sample_key).parts)])
    print(f"\nPrepared {len(outputs)} segmentation input(s).")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
