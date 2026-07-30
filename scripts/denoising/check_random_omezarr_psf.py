r"""Check the fixed four-stack 3D cohort against the reusable master PSFs.

Without ``--zarr``, one mapped source stack from each acquisition directory is
checked. Each stack is sampled at its configured U-Net target slice. Optional
Richardson-Lucy testing applies the same target-specific QC as
``deconvolve_3d_v2.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare  # noqa: E402
from PFT.core_prog_parts.denoising.deconvolution_no_fuji import (  # noqa: E402
    deconvolve_omezarr_3ch_to_omezarr_skimage,
)
from PFT.core_prog_parts.denoising.psf_creator import (  # noqa: E402
    validate_stack_against_master_psfs,
)
from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    check_3d_sample,
    configured_test_zarrs,
    target_slice_for_volume,
    volume_key,
    write_readiness_report,
)


def _find_zarrs(root: Path) -> list[Path]:
    """Return source image stores, excluding derived mask/deconvolution stores."""
    root = root.expanduser().resolve()
    if root.name == "image.ome.zarr" and root.is_dir():
        return [root]

    output: list[Path] = []
    for path in root.rglob("image.ome.zarr"):
        if not path.is_dir():
            continue
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts.intersection({"deconv", "segmentation", "masks", "predictions"}):
            continue
        output.append(path.resolve())
    return sorted(set(output))


def _parse_slices(value: str) -> tuple[int, ...]:
    slices = tuple(sorted({int(item) for item in value.replace(",", " ").split()}))
    if not slices or any(item < 1 for item in slices):
        raise ValueError("Sample slices must be positive one-based indices")
    return slices


def _selected_level0_array(zarr_dir: Path) -> tuple[zarr.Array, dict[str, Any]]:
    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=0)
    axes = str(meta.get("axes") or "").lower()
    if axes != "czyx":
        raise ValueError(f"Expected level-0 CZYX data, got axes={axes!r}")
    root = zarr.open_group(str(zarr_dir), mode="r")
    array_path = str(meta.get("array_path") or "0")
    array = root[array_path]
    if array.ndim != 4:
        raise ValueError(f"Expected four-dimensional CZYX data, got shape={array.shape}")
    if int(array.shape[0]) != 3:
        raise ValueError(f"Expected three fluorescence channels, got C={array.shape[0]}")
    if any(int(size) <= 0 for size in array.shape):
        raise ValueError(f"OME-Zarr contains an empty dimension: shape={array.shape}")
    return array, meta


def _sample_data_statistics(
    array: zarr.Array,
    *,
    slices_1based: Sequence[int],
    target_xy_samples: int = 256,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Read sparse planes and calculate channel statistics without full loading."""
    z_size = int(array.shape[1])
    valid_slices = tuple(value for value in slices_1based if 1 <= value <= z_size)
    if not valid_slices:
        valid_slices = (max(1, (z_size + 1) // 2),)

    y_size, x_size = int(array.shape[2]), int(array.shape[3])
    y_step = max(1, y_size // target_xy_samples)
    x_step = max(1, x_size // target_xy_samples)

    channel_records: list[dict[str, Any]] = []
    for channel_index in range(int(array.shape[0])):
        pieces: list[np.ndarray] = []
        for slice_1based in valid_slices:
            plane = np.asarray(
                array[channel_index, slice_1based - 1, ::y_step, ::x_step],
                dtype=np.float32,
            )
            pieces.append(plane.reshape(-1))
        values = np.concatenate(pieces)
        finite = np.isfinite(values)
        finite_values = values[finite]
        if finite_values.size == 0:
            raise ValueError(f"Channel {channel_index} has no finite sampled values")
        record = {
            "channel_index": channel_index,
            "sampled_voxels": int(values.size),
            "finite_fraction": float(finite.mean()),
            "nonzero_fraction": float(np.count_nonzero(finite_values) / finite_values.size),
            "minimum": float(np.min(finite_values)),
            "maximum": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values, dtype=np.float64)),
            "standard_deviation": float(np.std(finite_values, dtype=np.float64)),
        }
        if record["finite_fraction"] < 1.0:
            raise ValueError(
                f"Channel {channel_index} contains non-finite sampled values: "
                f"finite_fraction={record['finite_fraction']:.6f}"
            )
        if record["maximum"] <= record["minimum"]:
            raise ValueError(
                f"Channel {channel_index} is constant in sampled planes: "
                f"min=max={record['minimum']}"
            )
        if record["nonzero_fraction"] == 0.0:
            raise ValueError(f"Channel {channel_index} contains no nonzero sampled data")
        channel_records.append(record)
    return channel_records, valid_slices


def _report_paths(output_root: Path, sample: str) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    safe_sample = "".join(character if character.isalnum() or character in "-_" else "_" for character in sample)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        output_root / f"{safe_sample}__random_psf_check__{stamp}.txt",
        output_root / f"{safe_sample}__random_psf_check__{stamp}.json",
    )


def _write_compatibility_report(
    *,
    txt_path: Path,
    json_path: Path,
    selected_zarr: Path,
    discovery_root: Path,
    candidate_count: int,
    seed: int | None,
    meta: dict[str, Any],
    sampled_slices: Sequence[int],
    data_statistics: Sequence[dict[str, Any]],
    matches: Sequence[Any],
    master_metadata: dict[str, Any],
    readiness_report_txt: Path,
    readiness_report_json: Path,
) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    mappings = [
        {
            "channel_index": int(match.channel_index),
            "channel_name": str(match.channel_name),
            "wavelength_nm": float(match.wavelength_nm),
            "wavelength_source": str(match.wavelength_source),
            "display_color": str(match.display_color),
            "display_color_hex": str(match.display_color_hex),
            "psf_path": str(match.psf_path),
        }
        for match in matches
    ]
    payload = {
        "schema": "pft-four-stack-omezarr-master-psf-check-v1",
        "generated_utc": generated,
        "status": "PASS",
        "selection": {
            "discovery_root": str(discovery_root),
            "candidate_count": int(candidate_count),
            "seed": seed,
            "selected_omezarr": str(selected_zarr),
        },
        "level": 0,
        "image_metadata": meta,
        "sampled_slices_1based": list(sampled_slices),
        "sampled_data_statistics": list(data_statistics),
        "master_psf_metadata_path": str(
            PROJECT_ROOT / "results" / "psf" / "master" / "psf_master_metadata.json"
        ),
        "master_psf_model": master_metadata.get("model"),
        "compatibility_warnings": list(master_metadata.get("compatibility_warnings") or []),
        "current_stack_optical_metadata": master_metadata.get("current_stack_optical_metadata"),
        "resolved_optical_metadata_for_compatibility": master_metadata.get(
            "resolved_optical_metadata_for_compatibility"
        ),
        "channel_psf_mapping": mappings,
        "raw_readiness_report_txt": str(readiness_report_txt),
        "raw_readiness_report_json": str(readiness_report_json),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    lines = [
        "PFT fixed-cohort OME-Zarr and master PSF compatibility report",
        "=" * 80,
        f"Generated (UTC): {generated}",
        "Status: PASS",
        f"Discovery root: {discovery_root}",
        f"Candidate OME-Zarr stores: {candidate_count}",
        "Selection mode: fixed four-stack cohort or explicit --zarr",
        f"Selected OME-Zarr: {selected_zarr}",
        "Selected pyramid level: 0",
        f"Axes: {meta.get('axes')}",
        f"Shape: {meta.get('shape')}",
        f"Voxel size (um): {meta.get('voxel_size_um')}",
        f"Sampled slices (1-based): {tuple(sampled_slices)}",
        f"Master PSF model: {master_metadata.get('model')}",
        "",
        "Metadata compatibility notes",
        "-" * 80,
    ]
    warnings = list(master_metadata.get("compatibility_warnings") or [])
    if warnings:
        lines.extend(f"WARNING: {warning}" for warning in warnings)
    else:
        lines.append("No master-reference fallbacks were required.")
    lines.extend([
        "",
        "Channel data samples",
        "-" * 80,
    ])
    for record in data_statistics:
        lines.append(
            "Channel {channel_index}: min={minimum:.6g}, max={maximum:.6g}, "
            "mean={mean:.6g}, SD={standard_deviation:.6g}, "
            "nonzero={nonzero_fraction:.4f}, finite={finite_fraction:.4f}".format(**record)
        )
    lines.extend(["", "Wavelength-based master PSF mapping", "-" * 80])
    for mapping in mappings:
        lines.append(
            f"Channel {mapping['channel_index']} ({mapping['channel_name']}): "
            f"{mapping['wavelength_nm']:.3f} nm [{mapping['display_color']}] -> "
            f"{mapping['psf_path']}"
        )
    lines.extend([
        "",
        f"Raw readiness report: {readiness_report_txt}",
        f"JSON report: {json_path}",
        "RESULT: this stack contains usable sampled data and is compatible with all master PSFs.",
    ])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check one configured source stack from each acquisition directory "
            "against the reusable master PSFs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "results" / "img" / "3d_data",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="Check one exact image.ome.zarr instead of the fixed four-stack cohort",
    )
    parser.add_argument("--model", choices=("BW", "GL", "RW"), default="BW")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "noise_analysis" / "3d" / "four_stack_psf_check",
    )
    parser.add_argument("--run-deconvolution", action="store_true")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--iters-blue", type=int, default=None)
    parser.add_argument("--iters-green", type=int, default=None)
    parser.add_argument("--iters-red", type=int, default=None)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--filter-epsilon", type=float, default=None)
    parser.add_argument("--pyramid-max-layer", type=int, default=2)
    parser.add_argument(
        "--deconv-out-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "deconv_four_stack_check",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    discovery_root = args.root.expanduser().resolve()
    if args.zarr is not None:
        selected = args.zarr.expanduser().resolve()
        if not selected.is_dir() or selected.name != "image.ome.zarr":
            raise FileNotFoundError(f"--zarr must point to image.ome.zarr: {selected}")
        selected_zarrs = [selected]
    else:
        selected_zarrs = configured_test_zarrs(discovery_root)

    channel_iterations = {
        color: value
        for color, value in (
            ("blue", args.iters_blue),
            ("green", args.iters_green),
            ("red", args.iters_red),
        )
        if value is not None
    }
    if args.iters < 1 or any(value < 1 for value in channel_iterations.values()):
        raise ValueError("All iteration counts must be at least 1")

    print("\nPFT four-stack OME-Zarr and master PSF check")
    print("=" * 72)
    print(f"Stacks:    {len(selected_zarrs)}")
    print(f"PSF model: {args.model}")

    failures = 0
    for index, selected_zarr in enumerate(selected_zarrs, start=1):
        sample = volume_key(selected_zarr, discovery_root)
        target = target_slice_for_volume(selected_zarr, discovery_root)
        slices = (target,)
        print(f"\n[{index}/{len(selected_zarrs)}] {sample} | target Z{target}")
        try:
            readiness = check_3d_sample(
                selected_zarr,
                level=0,
                require_masks=False,
                training_slices_1based=slices,
            )
            readiness_root = args.report_root / "raw_readiness"
            readiness_txt, readiness_json = write_readiness_report(readiness, readiness_root)
            if not readiness.passed:
                raise ValueError(f"Raw-data readiness failed: {readiness_txt}")

            array, meta = _selected_level0_array(selected_zarr)
            data_statistics, sampled_slices = _sample_data_statistics(array, slices_1based=slices)
            matches, master_metadata = validate_stack_against_master_psfs(
                project_root=PROJECT_ROOT,
                zarr_dir=selected_zarr,
                level=0,
                model=args.model,
                normalize_psfs=True,
            )
            txt_path, json_path = _report_paths(args.report_root, sample)
            _write_compatibility_report(
                txt_path=txt_path,
                json_path=json_path,
                selected_zarr=selected_zarr,
                discovery_root=discovery_root,
                candidate_count=len(selected_zarrs),
                seed=None,
                meta=meta,
                sampled_slices=sampled_slices,
                data_statistics=data_statistics,
                matches=matches,
                master_metadata=master_metadata,
                readiness_report_txt=readiness_txt,
                readiness_report_json=readiness_json,
            )
            print(f"PASS: {txt_path}")

            if args.run_deconvolution:
                result = deconvolve_omezarr_3ch_to_omezarr_skimage(
                    in_omezarr=selected_zarr,
                    out_root=args.deconv_out_root,
                    model=args.model,
                    iters=args.iters,
                    channel_iterations=channel_iterations,
                    background=args.background,
                    level=0,
                    overwrite=not args.no_overwrite,
                    clip=False,
                    filter_epsilon=args.filter_epsilon,
                    pyramid_max_layer=args.pyramid_max_layer,
                    preview_slices_1based=slices,
                )
                print(f"Deconvolved: {result.out_zarr}")
                print(f"QC:          {result.preview_dir}")
        except Exception as error:
            failures += 1
            print(f"FAIL: {type(error).__name__}: {error}")

    print(f"\nChecked {len(selected_zarrs)} stack(s); failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
