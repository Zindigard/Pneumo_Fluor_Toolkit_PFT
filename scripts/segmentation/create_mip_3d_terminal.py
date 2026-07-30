"""Create raw or deconvolved 2D MIPs from the configured 3D PFT stacks.

Interactive mode allows selection of:

1. raw unmasked MIP for purely 2D segmentation;
2. raw MIP followed by the existing 2.5D U-Net mask;
3. BW Richardson-Lucy MIP without masking;
4. BW Richardson-Lucy MIP followed by configurable outside-mask suppression.

One configured source or all configured sources can be processed. The final
quantitative product is saved as CYX ``image.ome.zarr`` and the QC figure is a
merged-RGB comparison against the configured raw target slice.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    TARGET_SLICE_BY_VOLUME_KEY,
    configured_test_zarrs,
    target_slice_for_volume,
    volume_key,
)
from PFT.core_prog_parts.segmentation.mip_3d_core import (  # noqa: E402
    DECONV_DEFAULT_ITERS_BLUE,
    DECONV_DEFAULT_ITERS_GREEN,
    DECONV_DEFAULT_ITERS_RED,
    MIP3DConfig,
    MIPMode,
    MODE_DIRECTORY_NAMES,
    create_mip_for_volume,
    deconvolution_iteration_tag,
    mode_output_directory_name,
    resolve_deconvolved_zarr,
    resolve_predicted_mask_zarr,
)

MODE_LABELS: dict[str, str] = {
    "raw_unmasked": "Raw 40-slice MIP, no mask",
    "raw_masked": "Raw 40-slice MIP, then apply existing 2.5D U-Net mask",
    "deconv_unmasked": "Deconvolved 40-slice MIP, no mask",
    "deconv_masked": (
        "Deconvolved 40-slice MIP, then suppress outside the existing 2.5D mask"
    ),
}


def _configured_raw_zarrs(raw_root: Path) -> list[Path]:
    """Resolve the 29 mapped raw volumes in stable mapping order."""
    output: list[Path] = []
    for key in TARGET_SLICE_BY_VOLUME_KEY:
        path = raw_root / Path(*key.split("/")) / "image.ome.zarr"
        if not path.is_dir():
            raise FileNotFoundError(f"Configured raw OME-Zarr is missing: {path}")
        output.append(path.resolve())
    return output


def _select_mode_interactively() -> MIPMode:
    print("\nSelect MIP product:")
    print("  [1] Raw MIP without mask")
    print("  [2] Raw MIP followed by existing 2.5D mask")
    print("  [3] Deconvolved MIP without mask")
    print("  [4] Deconvolved MIP followed by outside-mask suppression")
    choice = input("Select number: ").strip()
    mapping: dict[str, MIPMode] = {
        "1": "raw_unmasked",
        "2": "raw_masked",
        "3": "deconv_unmasked",
        "4": "deconv_masked",
    }
    if choice not in mapping:
        raise ValueError("Expected MIP mode selection 1, 2, 3, or 4")
    return mapping[choice]


def _select_inputs_interactively(paths: Sequence[Path], raw_root: Path) -> list[Path]:
    print("\nProcess:")
    print("  [1] One configured image")
    print("  [2] Same four representative test images used for deconvolution")
    print("  [3] All configured images")
    scope = input("Select number: ").strip()
    if scope == "2":
        return configured_test_zarrs(raw_root)
    if scope == "3":
        return list(paths)
    if scope != "1":
        raise ValueError("Expected scope selection 1, 2, or 3")

    print("\nConfigured 3D images:")
    print("  [0] Exit")
    for index, path in enumerate(paths, start=1):
        target = target_slice_for_volume(path, image_root=raw_root)
        print(f"  [{index}] {volume_key(path, raw_root)} | target Z{target}")
    value = input("Select number: ").strip()
    selected = int(value)
    if selected == 0:
        return []
    if not 1 <= selected <= len(paths):
        raise ValueError(f"Selection must be between 0 and {len(paths)}")
    return [paths[selected - 1]]


def _write_summary(rows: list[dict[str, str]], output_root: Path, mode: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"mip_{mode}_summary.csv"
    fieldnames = [
        "generated_utc",
        "status",
        "mode",
        "sample",
        "target_slice_1based",
        "raw_zarr",
        "projection_source_zarr",
        "mask_zarr",
        "output_zarr",
        "qc_png",
        "snr_csv",
        "snr_json",
        "output_dtype",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 40-plane 2D MIP from raw or selected BW-deconvolved 3D "
            "OME-Zarr, optionally followed by configurable outside-mask suppression."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_DIRECTORY_NAMES),
        default=None,
        help="MIP product; omitted opens an interactive selector",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="Process one exact raw image.ome.zarr",
    )
    parser.add_argument(
        "--test-four",
        action="store_true",
        help="Process the same four representative images used for deconvolution testing",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all 29 configured raw volumes",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "img" / "3d_data",
    )
    parser.add_argument(
        "--deconv-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "deconv",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "U-net" / "3d_25d",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "mip_2d",
    )
    parser.add_argument("--level", type=int, choices=(0,), default=0)
    parser.add_argument(
        "--scale-bar-um",
        type=float,
        default=2.0,
        help="Physical scale-bar length drawn on every QC panel",
    )
    parser.add_argument(
        "--iters-blue",
        type=int,
        default=DECONV_DEFAULT_ITERS_BLUE,
        help="Richardson-Lucy iterations for the 405 nm/blue channel",
    )
    parser.add_argument(
        "--iters-green",
        type=int,
        default=DECONV_DEFAULT_ITERS_GREEN,
        help="Richardson-Lucy iterations for the 488 nm/green channel",
    )
    parser.add_argument(
        "--iters-red",
        type=int,
        default=DECONV_DEFAULT_ITERS_RED,
        help="Richardson-Lucy iterations for the 561 nm/red channel",
    )
    parser.add_argument(
        "--outside-suppression-percent",
        type=float,
        default=100.0,
        help=(
            "Percent intensity removed outside the U-Net mask; 99.8 retains 0.2%, "
            "while 100 sets outside pixels to zero"
        ),
    )
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    selected_scope_count = sum(
        (args.zarr is not None, bool(args.test_four), bool(args.all))
    )
    if selected_scope_count > 1:
        parser.error("Use only one of --zarr, --test-four, or --all")

    raw_root = args.raw_root.expanduser().resolve()
    deconv_root = args.deconv_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    mode: MIPMode = args.mode or _select_mode_interactively()

    configured = _configured_raw_zarrs(raw_root)
    if args.zarr is not None:
        selected = [args.zarr.expanduser().resolve()]
        # Resolve the target immediately to reject non-configured paths.
        target_slice_for_volume(selected[0], image_root=raw_root)
    elif args.test_four:
        selected = configured_test_zarrs(raw_root)
    elif args.all:
        selected = configured
    else:
        selected = _select_inputs_interactively(configured, raw_root)
    if not selected:
        print("No MIP was created.")
        return 0

    if args.iters_blue < 1 or args.iters_green < 1 or args.iters_red < 1:
        parser.error("All deconvolution iteration counts must be at least 1")
    if not 0.0 <= args.outside_suppression_percent <= 100.0:
        parser.error("--outside-suppression-percent must be between 0 and 100")

    preview_config = MIP3DConfig(
        project_root=PROJECT_ROOT,
        mode=mode,
        output_root=output_root,
        deconv_iters_blue=args.iters_blue,
        deconv_iters_green=args.iters_green,
        deconv_iters_red=args.iters_red,
        outside_suppression_percent=args.outside_suppression_percent,
    )
    output_directory_name = mode_output_directory_name(preview_config)
    iteration_tag = deconvolution_iteration_tag(
        args.iters_blue, args.iters_green, args.iters_red
    )
    is_masked = mode in ("raw_masked", "deconv_masked")
    is_deconvolved = mode in ("deconv_unmasked", "deconv_masked")

    print("\nPFT 3D to 2D maximum-intensity projection")
    print("=" * 76)
    print(f"Mode:                 {MODE_LABELS[mode]}")
    print(f"Input stacks:         {len(selected)}")
    print("Projected Z range:    Z1-Z40")
    print("Saved axes:           CYX")
    print("Stored normalization: NONE")
    if is_masked:
        print(
            f"Outside suppression:  {args.outside_suppression_percent:g}% "
            f"(retains {100.0 - args.outside_suppression_percent:g}%)"
        )
    else:
        print("Outside suppression:  not applied")
    print(
        f"Deconvolution:        BW {iteration_tag}"
        if is_deconvolved
        else "Deconvolution:        not used"
    )
    print(f"QC scale bars:        {args.scale_bar_um:g} µm on every panel")
    print(
        "SNR comparison:       raw target vs pre-mask and suppressed MIP, per channel"
        if is_masked
        else "SNR comparison:       not calculated without a mask"
    )
    print(f"Output root:          {output_root / output_directory_name}")

    rows: list[dict[str, str]] = []
    failures = 0
    skipped = 0
    for index, raw_zarr in enumerate(selected, start=1):
        sample = volume_key(raw_zarr, raw_root)
        target = target_slice_for_volume(raw_zarr, image_root=raw_root)
        print(f"\n[{index}/{len(selected)}] {sample}")
        print(f"Raw comparison and mask target: Z{target}")

        # In batch mode, missing deconvolution or mask products are explicit
        # skips so all available samples can still be completed.
        try:
            if mode in ("deconv_unmasked", "deconv_masked"):
                resolve_deconvolved_zarr(
                    raw_zarr,
                    deconv_root,
                    iters_blue=args.iters_blue,
                    iters_green=args.iters_green,
                    iters_red=args.iters_red,
                )
            if mode in ("raw_masked", "deconv_masked"):
                resolve_predicted_mask_zarr(raw_zarr, mask_root)
        except FileNotFoundError as error:
            if len(selected) == 1:
                raise
            skipped += 1
            rows.append({
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "status": "SKIP",
                "mode": mode,
                "sample": sample,
                "target_slice_1based": str(target),
                "raw_zarr": str(raw_zarr),
                "projection_source_zarr": "",
                "mask_zarr": "",
                "output_zarr": "",
                "qc_png": "",
                "snr_csv": "",
                "snr_json": "",
                "output_dtype": "",
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"SKIP: {error}")
            continue

        try:
            result = create_mip_for_volume(
                MIP3DConfig(
                    project_root=PROJECT_ROOT,
                    raw_zarr=raw_zarr,
                    mode=mode,
                    raw_root=raw_root,
                    deconv_root=deconv_root,
                    mask_root=mask_root,
                    output_root=output_root,
                    level=args.level,
                    overwrite=not args.no_overwrite,
                    scale_bar_um=args.scale_bar_um,
                    deconv_iters_blue=args.iters_blue,
                    deconv_iters_green=args.iters_green,
                    deconv_iters_red=args.iters_red,
                    outside_suppression_percent=args.outside_suppression_percent,
                )
            )
            rows.append({
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "status": "PASS",
                "mode": mode,
                "sample": sample,
                "target_slice_1based": str(target),
                "raw_zarr": str(result.raw_zarr),
                "projection_source_zarr": str(result.projection_source_zarr),
                "mask_zarr": str(result.mask_zarr) if result.mask_zarr else "",
                "output_zarr": str(result.output_zarr),
                "qc_png": str(result.qc_png),
                "snr_csv": str(result.snr_csv) if result.snr_csv else "",
                "snr_json": str(result.snr_json) if result.snr_json else "",
                "output_dtype": result.output_dtype,
                "error": "",
            })
            print(f"PASS OME-Zarr: {result.output_zarr}")
            print(f"QC merged RGB: {result.qc_png}")
            print(f"Stored dtype: {result.output_dtype}")
            if result.snr_csv is not None:
                print(f"SNR comparison: {result.snr_csv}")
        except Exception as error:
            failures += 1
            rows.append({
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "status": "FAIL",
                "mode": mode,
                "sample": sample,
                "target_slice_1based": str(target),
                "raw_zarr": str(raw_zarr),
                "projection_source_zarr": "",
                "mask_zarr": "",
                "output_zarr": "",
                "qc_png": "",
                "snr_csv": "",
                "snr_json": "",
                "output_dtype": "",
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"FAIL: {type(error).__name__}: {error}")
            if args.stop_on_error:
                break

    summary = _write_summary(rows, output_root, output_directory_name)
    passed = sum(row["status"] == "PASS" for row in rows)
    print("\nMIP batch completed")
    print(f"Passed: {passed} | Skipped: {skipped} | Failed: {failures}")
    print(f"Summary: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
