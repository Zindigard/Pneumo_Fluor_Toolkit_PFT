"""
Run 2D U-Net inference and save a non-normalized enhanced OME-Zarr.

The predicted mask is applied to the original filtered image, not to the
normalized network tensor. By default, pixels outside the mask are depleted by
98% and retain 2% of their original intensity. The saved OME-Zarr is then
reloaded for ROI-SNR calculation.
"""

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

from PFT.core_prog_parts.segmentation.unet_run_core import (  # noqa: E402
    UNetRunConfig,
    default_input_root,
    default_mask_root,
    default_model_path,
    default_out_root,
    default_raw_root,
    find_omezarr_dirs,
    run_dataset,
)
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import DATASETS_2D  # noqa: E402


def _choose_dataset() -> str:
    print("\nChoose 2D U-Net model:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    return "2d_wga_dapi" if input("Choose number [1]: ").strip() == "2" else "2d_time"


def _choose_mode() -> str:
    print("\nChoose inference mode:")
    print("  1) one sample")
    print("  2) selected samples")
    print("  3) all samples and every frame in each stack")
    answer = input("Choose number [3]: ").strip()
    if answer == "1":
        return "one"
    if answer == "2":
        return "selected"
    return "all"


def _choose_indices(paths: list[Path], multiple: bool) -> list[Path]:
    for index, path in enumerate(paths):
        print(f"  [{index}] {path.parent.name} | {path}")
    prompt = "Enter comma-separated indices: " if multiple else "Enter index: "
    answer = input(prompt).strip()
    values = [int(value.strip()) for value in answer.split(",") if value.strip()]
    if not multiple and len(values) != 1:
        raise ValueError("Choose exactly one index")
    selected: list[Path] = []
    for index in values:
        if index < 0 or index >= len(paths):
            raise IndexError(f"Index outside range: {index}")
        selected.append(paths[index])
    return selected


def _normalize_input_path(path: Path) -> Path:
    path = Path(path)
    if path.name == "image.ome.zarr" and path.is_dir():
        return path
    candidate = path / "image.ome.zarr"
    if candidate.is_dir():
        return candidate
    if path.is_dir() and path.name.endswith(".ome.zarr"):
        return path
    raise FileNotFoundError(f"Input is not an OME-Zarr image or sample directory: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 2D U-Net foreground inference and calculate IoU and ROI SNR."
    )
    parser.add_argument("--dataset", choices=DATASETS_2D)
    parser.add_argument("--mode", choices=("one", "selected", "all"))
    parser.add_argument(
        "--input",
        type=Path,
        help="Direct image.ome.zarr or sample directory for --mode one.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Sample directory names for --mode selected.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="Zero-based indices printed by --list-inputs or interactive mode.",
    )
    parser.add_argument("--list-inputs", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument(
        "--outside-mask-depletion",
        type=float,
        default=0.98,
        help=(
            "Fraction removed outside the predicted mask. Default 0.98 removes "
            "98%% and leaves 2%% of the original non-normalized intensity. "
            "Use 1.0 for complete removal or 0.0 for no attenuation."
        ),
    )
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument(
        "--strict-metrics",
        action="store_true",
        help=(
            "Fail when any selected sample lacks a reference mask. By default, "
            "such samples are still processed and saved, while metrics are skipped."
        ),
    )
    parser.add_argument("--no-probability", action="store_true")
    parser.add_argument("--no-foreground-image", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset is None:
        if args.non_interactive:
            raise SystemExit("--dataset is required with --non-interactive")
        dataset = _choose_dataset()
    else:
        dataset = args.dataset

    cfg = UNetRunConfig(project_root=PROJECT_ROOT, dataset=dataset)
    cfg.model_path = Path(args.model_path or default_model_path(PROJECT_ROOT, dataset))
    cfg.input_root = Path(args.input_root or default_input_root(PROJECT_ROOT, dataset))
    cfg.out_root = Path(args.output_root or default_out_root(PROJECT_ROOT, dataset))
    cfg.raw_root = Path(args.raw_root or default_raw_root(PROJECT_ROOT, dataset))
    cfg.mask_root = Path(args.mask_root or default_mask_root(PROJECT_ROOT, dataset))
    cfg.level = args.level
    cfg.patch = args.patch
    cfg.threshold = args.threshold
    cfg.predict_batch_size = args.predict_batch_size
    cfg.outside_mask_depletion = args.outside_mask_depletion
    cfg.compute_metrics = not args.no_metrics
    cfg.strict_metrics = args.strict_metrics
    cfg.save_probability = not args.no_probability
    cfg.save_foreground_image = not args.no_foreground_image

    available = find_omezarr_dirs(cfg.input_root)
    if args.list_inputs:
        for index, path in enumerate(available):
            print(f"[{index}] {path.parent.name} | {path}")
        return 0

    mode = args.mode
    if mode is None:
        if args.non_interactive:
            raise SystemExit("--mode is required with --non-interactive")
        mode = _choose_mode()

    selected: list[Path] | None = None
    process_all = mode == "all"
    if mode == "one":
        if args.input is not None:
            selected = [_normalize_input_path(args.input)]
        elif args.indices:
            if len(args.indices) != 1:
                raise SystemExit("--mode one accepts exactly one --indices value")
            selected = [available[args.indices[0]]]
        elif args.non_interactive:
            raise SystemExit("--mode one requires --input or one --indices value")
        else:
            selected = _choose_indices(available, multiple=False)
    elif mode == "selected":
        if args.samples:
            by_name = {path.parent.name: path for path in available}
            missing = [name for name in args.samples if name not in by_name]
            if missing:
                raise SystemExit(f"Unknown sample name(s): {', '.join(missing)}")
            selected = [by_name[name] for name in args.samples]
        elif args.indices:
            selected = [available[index] for index in args.indices]
        elif args.non_interactive:
            raise SystemExit("--mode selected requires --samples or --indices")
        else:
            selected = _choose_indices(available, multiple=True)

    print("\n=== 2D U-NET INFERENCE CONFIGURATION ===")
    print(f"Dataset:              {cfg.dataset}")
    print(f"Mode:                 {mode}")
    print(f"Model:                {cfg.model_path}")
    print(f"Filtered input root:  {cfg.input_root}")
    print(f"Raw root:             {cfg.raw_root}")
    print(f"Reference masks:      {cfg.mask_root}")
    print(f"Output root:          {cfg.out_root}")
    print("Runtime normalization: complete image, per channel, P1-P99.8 -> [0,1]")
    print(f"Threshold:            {cfg.threshold}")
    print(
        f"Outside-mask change:  {100.0 * cfg.outside_mask_depletion:.1f}% depleted, "
        f"{100.0 * (1.0 - cfg.outside_mask_depletion):.1f}% retained"
    )
    print("Saved intensity scale: original filtered dtype and values; no normalization")
    print("SNR after source:     reloaded foreground_filtered.ome.zarr")
    print(f"Compute IoU and SNR:  {cfg.compute_metrics}")
    print(f"Strict missing masks: {cfg.strict_metrics}")
    print(
        "QC visualization:     all 2d_time panels use blue fluorescence; "
        "2d_wga_dapi retains its channel colours"
    )

    outputs = run_dataset(cfg, process_all=process_all, selected_zarrs=selected)
    print("\nInference finished:")
    for output in outputs:
        print(f"  {output.sample}")
        print(f"    mask:       {output.mask_zarr}")
        if output.foreground_zarr:
            print(f"    enhanced:   {output.foreground_zarr}")
        print(f"    QC preview: {output.preview_png}")
        if output.metrics_json:
            print(f"    metrics:    {output.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
