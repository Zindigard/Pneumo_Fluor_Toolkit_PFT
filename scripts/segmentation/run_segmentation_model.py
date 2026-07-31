"""
Run Cellpose, Omnipose, or StarDist on prepared PFT segmentation inputs.

The script reads only ``segmentation_input.ome.zarr`` files. These files are
already normalized to float32 [0, 1], so model-side normalization is explicitly
disabled.

Run the Cellpose ``cpsam`` initial model on all 2d_time images::

    python scripts/segmentation/run_segmentation_model.py `
        --family cellpose `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --model cpsam `
        --mode all

Run Omnipose on one deconvolved MIP::

    python scripts/segmentation/run_segmentation_model.py `
        --family omnipose `
        --dataset 3d_mip `
        --source-mode deconv_masked `
        --model-name bact_fluor_omni `
        --mode one

Run a trained StarDist model folder::

    python scripts/segmentation/run_segmentation_model.py `
        --family stardist `
        --dataset 2d_wga_dapi `
        --source-mode filtered_unet `
        --model "models/segmentation/stardist/2d_wga_dapi/filtered_unet/my_run"
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
    PredictionConfig,
    list_prepared_inputs,
    run_prediction_dataset,
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


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value.lower() in {"none", "auto"}:
        return None
    return float(value)


def main(default_family: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an instance-segmentation model on prepared float32 OME-Zarr inputs.",
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
    parser.add_argument("--mode", choices=("one", "all"), default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Built-in model name or path to a trained model; family-specific defaults are used when omitted",
    )
    parser.add_argument("--model-name", default=None, help="Output/model display name")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--diameter", default="none")
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prob-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    args = parser.parse_args()

    family = args.family or _choose("Choose model family", MODEL_FAMILIES)
    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is invalid for {dataset!r}: {valid_modes}")

    items = list_prepared_inputs(PROJECT_ROOT, dataset, source_mode)
    if not items:
        raise FileNotFoundError(
            "No prepared segmentation inputs were found. Run prepare_segmentation_inputs.py first."
        )
    run_mode = args.mode or _choose("Process one sample or all samples", ("one", "all"))
    selected: set[str] | None = None
    if run_mode == "one":
        keys = [item.sample_key for item in items]
        sample = args.sample or _choose("Choose sample", keys)
        if sample not in keys:
            parser.error(f"Unknown sample key: {sample}")
        selected = {sample}

    cfg = PredictionConfig(
        project_root=PROJECT_ROOT,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        model=args.model,
        model_name=args.model_name,
        gpu=not args.cpu,
        selected_sample_keys=selected,
        diameter=_parse_optional_float(args.diameter),
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        mask_threshold=args.mask_threshold,
        min_size=args.min_size,
        batch_size=args.batch_size,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
    )

    print("\nPFT instance-segmentation prediction")
    print("=" * 72)
    print(f"Family:              {family}")
    print(f"Dataset:             {dataset}")
    print(f"Source mode:         {source_mode}")
    print(f"Samples:             {len(items) if selected is None else 1}")
    print("Input contract:      prepared float32 [0,1]")
    print("Model normalization: DISABLED")

    output = run_prediction_dataset(cfg)
    print(f"\nSaved predictions to:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
