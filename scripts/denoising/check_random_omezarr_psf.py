r"""Randomly verify one 3D OME-Zarr against the reusable master PSF set.

The default mode is a fast, non-destructive preflight check. The script:

1. discovers all ``image.ome.zarr`` stores under the configured 3D-data root;
2. selects one store randomly, with an optional reproducible random seed;
3. confirms that level 0 contains a finite, non-constant CZYX volume;
4. samples the planned training slices without loading the complete volume;
5. validates voxel sampling, optics, wavelengths, spatial dimensions, and all
   three master PSFs against ``results/psf/master``;
6. records the wavelength-to-color-to-PSF mapping in TXT and JSON reports.

Use ``--run-deconvolution`` only when a full test run is required. It calls the
same raw-intensity-preserving Richardson-Lucy function as
``deconvolve_3d_v2.py``. Full level-0 deconvolution can require substantial RAM
and processing time; the default compatibility check does not deconvolve.

Examples
--------
Fast random compatibility check::

    & $PY scripts\denoising\check_random_omezarr_psf.py

Repeatable random selection::

    & $PY scripts\denoising\check_random_omezarr_psf.py --seed 42

Check a specified store instead of selecting randomly::

    & $PY scripts\denoising\check_random_omezarr_psf.py `
      --zarr "D:\\...\\image.ome.zarr"

Run full low-iteration Richardson-Lucy after the successful check::

    & $PY scripts\denoising\check_random_omezarr_psf.py `
      --seed 42 --run-deconvolution --iters-blue 4 --iters-green 5 --iters-red 3
"""

from __future__ import annotations

import argparse
import json
import random
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
    DEFAULT_TRAINING_SLICES_1BASED,
    deconvolve_omezarr_3ch_to_omezarr_skimage,
)
from PFT.core_prog_parts.denoising.psf_creator import (  # noqa: E402
    validate_stack_against_master_psfs,
)
from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    check_3d_sample,
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
        "schema": "pft-random-omezarr-master-psf-check-v1",
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
        "PFT random OME-Zarr and master PSF compatibility report",
        "=" * 80,
        f"Generated (UTC): {generated}",
        "Status: PASS",
        f"Discovery root: {discovery_root}",
        f"Candidate OME-Zarr stores: {candidate_count}",
        f"Random seed: {seed if seed is not None else 'system random'}",
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
        description="Randomly check one source 3D OME-Zarr against the reusable master PSFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "results" / "img" / "3d_data",
        help="Root containing source image.ome.zarr stores",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="Check this exact image.ome.zarr instead of random selection",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random seed")
    parser.add_argument("--model", choices=("BW", "GL", "RW"), default="BW")
    parser.add_argument(
        "--sample-slices",
        default=",".join(str(value) for value in DEFAULT_TRAINING_SLICES_1BASED),
        help="One-based Z slices sampled during the quick data check",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "noise_analysis" / "3d" / "random_psf_check",
    )
    parser.add_argument(
        "--run-deconvolution",
        action="store_true",
        help="After a successful check, run full level-0 Richardson-Lucy deconvolution",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Default iteration count for the optional full test",
    )
    parser.add_argument("--iters-blue", type=int, default=None, help="Optional 405 nm blue-channel iteration count")
    parser.add_argument("--iters-green", type=int, default=None, help="Optional 488 nm green-channel iteration count")
    parser.add_argument("--iters-red", type=int, default=None, help="Optional 561 nm red-channel iteration count")
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--filter-epsilon", type=float, default=None)
    parser.add_argument("--pyramid-max-layer", type=int, default=2)
    parser.add_argument(
        "--deconv-out-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "deconv_random_check",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    discovery_root = args.root.expanduser().resolve()
    if args.zarr is not None:
        selected_zarr = args.zarr.expanduser().resolve()
        if not selected_zarr.is_dir():
            raise FileNotFoundError(selected_zarr)
        candidates = _find_zarrs(selected_zarr)
        if candidates != [selected_zarr]:
            raise ValueError(f"--zarr must point to an image.ome.zarr directory: {selected_zarr}")
        candidate_count = 1
    else:
        candidates = _find_zarrs(discovery_root)
        if not candidates:
            raise FileNotFoundError(f"No source image.ome.zarr stores found under {discovery_root}")
        selected_zarr = random.Random(args.seed).choice(candidates)
        candidate_count = len(candidates)

    slices = _parse_slices(args.sample_slices)
    channel_iterations = {
        color: value
        for color, value in (
            ("blue", args.iters_blue),
            ("green", args.iters_green),
            ("red", args.iters_red),
        )
        if value is not None
    }
    if any(value < 1 for value in channel_iterations.values()):
        raise ValueError("All per-channel iteration counts must be at least 1")
    print("\nPFT random OME-Zarr and master PSF check")
    print("=" * 72)
    print(f"Available source stores: {candidate_count}")
    print(f"Random seed:            {args.seed if args.seed is not None else 'system random'}")
    print(f"Selected store:         {selected_zarr}")
    print("Pyramid level:          0")
    print(f"PSF model:              {args.model}")

    readiness = check_3d_sample(
        selected_zarr,
        level=0,
        require_masks=False,
        training_slices_1based=slices,
    )
    readiness_root = args.report_root / "raw_readiness"
    readiness_txt, readiness_json = write_readiness_report(readiness, readiness_root)
    if not readiness.passed:
        raise ValueError(f"Randomly selected OME-Zarr failed raw-data readiness checks: {readiness_txt}")

    array, meta = _selected_level0_array(selected_zarr)
    data_statistics, sampled_slices = _sample_data_statistics(array, slices_1based=slices)
    matches, master_metadata = validate_stack_against_master_psfs(
        project_root=PROJECT_ROOT,
        zarr_dir=selected_zarr,
        level=0,
        model=args.model,
        normalize_psfs=True,
    )

    txt_path, json_path = _report_paths(args.report_root, selected_zarr.parent.name)
    _write_compatibility_report(
        txt_path=txt_path,
        json_path=json_path,
        selected_zarr=selected_zarr,
        discovery_root=discovery_root,
        candidate_count=candidate_count,
        seed=args.seed,
        meta=meta,
        sampled_slices=sampled_slices,
        data_statistics=data_statistics,
        matches=matches,
        master_metadata=master_metadata,
        readiness_report_txt=readiness_txt,
        readiness_report_json=readiness_json,
    )

    warnings = list(master_metadata.get("compatibility_warnings") or [])
    print("\nPASS")
    if warnings:
        print("Metadata note: fixed master-reference values were used for absent fields:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"Compatibility TXT: {txt_path}")
    print(f"Compatibility JSON:{json_path}")
    for match in matches:
        print(
            f"  C{match.channel_index} {match.channel_name}: "
            f"{match.wavelength_nm:g} nm ({match.display_color}) -> {match.psf_path.name}"
        )

    if args.run_deconvolution:
        print("\nStarting optional full Richardson-Lucy test...")
        print(f"Default iterations: {args.iters}")
        print(
            "Per-channel iterations: "
            + (
                ", ".join(f"{color}={value}" for color, value in channel_iterations.items())
                if channel_iterations
                else "none; default used for all channels"
            )
        )
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
        print(f"Deconvolved OME-Zarr: {result.out_zarr}")
        print(f"Deconvolution report: {result.report_txt}")
    else:
        print("No deconvolution was run. Add --run-deconvolution for a full test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
