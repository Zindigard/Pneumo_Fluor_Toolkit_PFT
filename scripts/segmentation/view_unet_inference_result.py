"""
Interactively inspect one 2D U-Net inference result in napari.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
DATASETS = ("2d_time", "2d_wga_dapi")
VIEWS = ("comparison", "all", "result")
EXPECTED_FILTER_PERCENTILE = {
    "2d_time": 97.5,
    "2d_wga_dapi": 98.0,
}


def find_project_root() -> Path:
    """Locate the repository root containing ``scripts`` and ``src/PFT``."""
    for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate project root containing scripts and src/PFT")


PROJECT_ROOT = find_project_root()
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from PFT.core_prog_parts.decoder_omezar import (  # noqa: E402
    load_ome_zarr,
    move_yx_to_last,
    normalize_axes,
)


def prompt_choice(label: str, options: Sequence[str]) -> str:
    """Request one numbered terminal selection."""
    if not options:
        raise ValueError(f"No options are available for: {label}")
    print(f"\n{label}")
    for index, option in enumerate(options, start=1):
        print(f"  {index:>3}. {option}")
    while True:
        answer = input("Select number: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            print("Enter a valid number.")
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print(f"Enter a number from 1 to {len(options)}.")


def filtered_root(dataset: str) -> Path:
    """Resolve the production intensity-preserving local-threshold folder."""
    base = (
        PROJECT_ROOT
        / "results"
        / "Filters"
        / "Local_high_threshold"
        / dataset
        / "intensity_preserved"
    )
    percentile = EXPECTED_FILTER_PERCENTILE[dataset]
    preferred = base / f"p{percentile:.1f}_k3f40_s370_k10f40"
    if preferred.is_dir():
        return preferred

    candidates = sorted(
        path for path in base.glob(f"p{percentile:.1f}_*") if path.is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No p{percentile:.1f} local-threshold folder found under: {base}"
        )
    raise RuntimeError(
        "Several matching local-threshold folders were found. "
        "Pass --filtered-root explicitly:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def discover_inference_roots(dataset: str) -> list[Path]:
    """Return completed inference roots matching one dataset."""
    base = PROJECT_ROOT / "results" / "U-net"
    if not base.is_dir():
        return []

    candidates: list[Path] = []
    for path in sorted(item for item in base.iterdir() if item.is_dir()):
        if path.name != dataset and not path.name.startswith(f"{dataset}_"):
            continue
        has_output = any(
            child.is_dir() and (child / "foreground_filtered.ome.zarr").is_dir()
            for child in path.iterdir()
        )
        if has_output:
            candidates.append(path)
    return candidates


def discover_samples(output_root: Path) -> dict[str, Path]:
    """Return completed sample folders containing the saved U-Net result."""
    if not output_root.is_dir():
        raise FileNotFoundError(f"Inference output root does not exist: {output_root}")
    samples = {
        sample_dir.name: sample_dir
        for sample_dir in sorted(output_root.iterdir())
        if sample_dir.is_dir()
        and (sample_dir / "foreground_filtered.ome.zarr").is_dir()
    }
    if not samples:
        raise FileNotFoundError(
            f"No <sample>/foreground_filtered.ome.zarr outputs found under: {output_root}"
        )
    return samples


def select_axis(
    array: np.ndarray,
    axes: str,
    axis: str,
    index: int,
) -> tuple[np.ndarray, str]:
    """Select one index and remove the corresponding axis label."""
    position = axes.index(axis)
    array = np.take(array, int(index), axis=position)
    return array, axes[:position] + axes[position + 1 :]


def load_hwc_frames(zarr_path: Path, dataset: str, level: int) -> list[np.ndarray]:
    """Load all frames as HWC arrays without changing stored intensity values."""
    array, axes = load_ome_zarr(zarr_path, level=level, as_numpy=True)
    array = np.asarray(array)
    axes = normalize_axes(axes)
    array, axes = move_yx_to_last(array, axes)

    for axis in tuple(axes):
        if axis in {"t", "c", "y", "x"}:
            continue
        position = axes.index(axis)
        size = int(array.shape[position])
        if size != 1:
            raise ValueError(
                f"Unsupported non-singleton axis {axis}={size} in {zarr_path}; "
                f"axes={axes}, shape={array.shape}"
            )
        array = np.take(array, 0, axis=position)
        axes = axes[:position] + axes[position + 1 :]

    time_count = int(array.shape[axes.index("t")]) if "t" in axes else 1
    frames: list[np.ndarray] = []
    for time_index in range(time_count):
        frame = array
        frame_axes = axes
        if "t" in frame_axes:
            frame, frame_axes = select_axis(frame, frame_axes, "t", time_index)

        if frame_axes == "yx":
            hwc = frame[..., None]
        elif frame_axes == "cyx":
            channel_axis = frame_axes.index("c")
            channel_count = int(frame.shape[channel_axis])
            if dataset == "2d_time":
                hwc = np.take(frame, 0, axis=channel_axis)[..., None]
            else:
                if channel_count < 2:
                    raise ValueError(
                        f"2d_wga_dapi requires at least two channels, found "
                        f"{channel_count}: {zarr_path}"
                    )
                hwc = np.stack(
                    [
                        np.take(frame, 0, axis=channel_axis),
                        np.take(frame, 1, axis=channel_axis),
                    ],
                    axis=-1,
                )
        else:
            raise ValueError(
                f"Unsupported OME-Zarr axes '{frame_axes}' for {zarr_path}"
            )
        frames.append(np.asarray(hwc))
    return frames


def normalize_image01(image: np.ndarray) -> np.ndarray:
    """Apply the exact complete-image, per-channel P1-P99.8 normalization."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected YX or YXC image, received shape={image.shape}")

    output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        plane = image[..., channel]
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            output[..., channel] = 0.0
            continue
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.8))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            output[..., channel] = 0.0
        else:
            output[..., channel] = np.clip(
                (plane - low) / (high - low + 1e-8),
                0.0,
                1.0,
            )
    return output


def normalize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize every complete frame independently, matching U-Net inference."""
    return [normalize_image01(frame) for frame in frames]


def stack_channel(frames: list[np.ndarray], channel: int) -> np.ndarray:
    """Convert HWC frame lists to YX or TYX napari arrays."""
    planes = [np.asarray(frame[..., channel]) for frame in frames]
    return planes[0] if len(planes) == 1 else np.stack(planes, axis=0)


def array_statistics(frames: list[np.ndarray]) -> tuple[str, tuple[int, ...], float, float]:
    """Return dtype, frame shape, finite minimum, and finite maximum."""
    sample = np.asarray(frames[0])
    finite_parts = [np.asarray(frame)[np.isfinite(frame)] for frame in frames]
    finite_parts = [part for part in finite_parts if part.size]
    if finite_parts:
        minimum = float(min(float(np.min(part)) for part in finite_parts))
        maximum = float(max(float(np.max(part)) for part in finite_parts))
    else:
        minimum = float("nan")
        maximum = float("nan")
    return str(sample.dtype), tuple(sample.shape), minimum, maximum


def add_source_layers(
    viewer,
    *,
    dataset: str,
    label: str,
    frames: list[np.ndarray],
    visible: bool,
    metadata: dict[str, object],
) -> None:
    """Add one single-channel or two-channel source using biological colours."""
    if dataset == "2d_time":
        viewer.add_image(
            stack_channel(frames, 0),
            name=label,
            colormap="blue",
            visible=visible,
            metadata=metadata,
        )
        return

    names = ("DAPI", "WGA")
    colormaps = ("blue", "green")
    for channel, (channel_name, colormap) in enumerate(zip(names, colormaps)):
        viewer.add_image(
            stack_channel(frames, channel),
            name=f"{label} | {channel_name}",
            colormap=colormap,
            visible=visible,
            blending="additive",
            metadata={**metadata, "channel": channel_name},
        )


def add_optional_mask_layers(viewer, sample_dir: Path, level: int) -> None:
    """Add predicted mask and probability layers when available."""
    mask_path = sample_dir / "pred_mask.ome.zarr"
    if mask_path.is_dir():
        mask_array, mask_axes = load_ome_zarr(mask_path, level=level, as_numpy=True)
        mask_array = np.asarray(mask_array)
        mask_axes = normalize_axes(mask_axes)
        mask_array = np.squeeze(mask_array)
        viewer.add_labels(
            mask_array.astype(np.uint8, copy=False),
            name="Predicted foreground mask",
            visible=False,
            opacity=0.45,
            metadata={"source_path": str(mask_path), "axes": mask_axes},
        )

    probability_path = sample_dir / "foreground_probability.ome.zarr"
    if probability_path.is_dir():
        probability_array, probability_axes = load_ome_zarr(
            probability_path, level=level, as_numpy=True
        )
        probability_array = np.squeeze(np.asarray(probability_array, dtype=np.float32))
        viewer.add_image(
            probability_array,
            name="Foreground probability",
            colormap="blue",
            contrast_limits=(0.0, 1.0),
            visible=False,
            metadata={
                "source_path": str(probability_path),
                "axes": normalize_axes(probability_axes),
            },
        )


def open_result(
    *,
    dataset: str,
    output_root: Path,
    sample: str,
    filtered_root_override: Path | None,
    level: int,
    view: str,
    grid: bool,
) -> None:
    """Load selected sources and open them in one napari viewer."""
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed in the selected Python environment."
        ) from exc

    sample_dir = output_root / sample
    result_path = sample_dir / "foreground_filtered.ome.zarr"
    raw_path = PROJECT_ROOT / "results" / "img" / dataset / sample / "image.ome.zarr"
    input_base = (
        filtered_root_override.expanduser().resolve()
        if filtered_root_override is not None
        else filtered_root(dataset)
    )
    filtered_path = input_base / sample / "image.ome.zarr"

    required = {
        "raw": raw_path,
        "filtered U-Net input": filtered_path,
        "U-Net foreground-filtered result": result_path,
    }
    for label, path in required.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label} OME-Zarr: {path}")

    raw_frames = load_hwc_frames(raw_path, dataset, level)
    filtered_frames = load_hwc_frames(filtered_path, dataset, level)
    result_frames = load_hwc_frames(result_path, dataset, 0)

    if not (
        len(raw_frames) == len(filtered_frames) == len(result_frames)
        and raw_frames[0].shape == filtered_frames[0].shape == result_frames[0].shape
    ):
        raise ValueError(
            "Raw, filtered, and U-Net result dimensions do not agree:\n"
            f"  raw:      {len(raw_frames)} frames, {raw_frames[0].shape}\n"
            f"  filtered: {len(filtered_frames)} frames, {filtered_frames[0].shape}\n"
            f"  result:   {len(result_frames)} frames, {result_frames[0].shape}"
        )

    normalized_input = normalize_frames(filtered_frames)
    normalized_result = normalize_frames(result_frames)

    print("\n=== U-NET RESULT VIEWER ===")
    print(f"Dataset:          {dataset}")
    print(f"Inference root:   {output_root}")
    print(f"Sample:           {sample}")
    print(f"Raw OME-Zarr:     {raw_path}")
    print(f"Filtered input:   {filtered_path}")
    print(f"Saved U-Net data: {result_path}")
    print("Normalization:    in-memory P1-P99.8 per complete frame and channel")
    print("Saved result:     original intensity scale; normalization is display-only")

    for label, frames in (
        ("raw", raw_frames),
        ("filtered input", filtered_frames),
        ("U-Net result", result_frames),
        ("normalized U-Net result", normalized_result),
    ):
        dtype, shape, minimum, maximum = array_statistics(frames)
        print(
            f"  {label:<24} dtype={dtype:<10} frame_shape={shape!s:<18} "
            f"min={minimum:.6g} max={maximum:.6g}"
        )

    viewer = napari.Viewer(title=f"PFT U-Net | {dataset} | {sample}")

    if view == "comparison":
        add_source_layers(
            viewer,
            dataset=dataset,
            label="1 Raw image",
            frames=raw_frames,
            visible=True,
            metadata={"source_path": str(raw_path), "normalized": False},
        )
        add_source_layers(
            viewer,
            dataset=dataset,
            label="2 Normalized U-Net input",
            frames=normalized_input,
            visible=True,
            metadata={
                "source_path": str(filtered_path),
                "normalized": True,
                "normalization": "P1-P99.8 per complete frame and channel",
            },
        )
        add_source_layers(
            viewer,
            dataset=dataset,
            label="3 Normalized U-Net result",
            frames=normalized_result,
            visible=True,
            metadata={
                "source_path": str(result_path),
                "normalized": True,
                "normalization": "P1-P99.8 per complete frame and channel",
            },
        )
    elif view == "result":
        add_source_layers(
            viewer,
            dataset=dataset,
            label="U-Net result | original scale",
            frames=result_frames,
            visible=False,
            metadata={"source_path": str(result_path), "normalized": False},
        )
        add_source_layers(
            viewer,
            dataset=dataset,
            label="U-Net result | normalized",
            frames=normalized_result,
            visible=True,
            metadata={
                "source_path": str(result_path),
                "normalized": True,
                "normalization": "P1-P99.8 per complete frame and channel",
            },
        )
    else:
        for label, frames, path, normalized, visible in (
            ("Raw image | original scale", raw_frames, raw_path, False, False),
            ("Filtered input | original scale", filtered_frames, filtered_path, False, False),
            ("Filtered input | normalized", normalized_input, filtered_path, True, True),
            ("U-Net result | original scale", result_frames, result_path, False, False),
            ("U-Net result | normalized", normalized_result, result_path, True, True),
        ):
            add_source_layers(
                viewer,
                dataset=dataset,
                label=label,
                frames=frames,
                visible=visible,
                metadata={
                    "source_path": str(path),
                    "normalized": normalized,
                    "normalization": (
                        "P1-P99.8 per complete frame and channel" if normalized else "none"
                    ),
                },
            )
        add_optional_mask_layers(viewer, sample_dir, level=0)

    if grid:
        viewer.grid.enabled = True

    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open one completed 2D U-Net inference result and compare the raw image, "
            "normalized network input, and normalized saved foreground-filtered image."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Inference run root containing <sample>/foreground_filtered.ome.zarr. "
            "When omitted, matching runs under results/U-net are listed."
        ),
    )
    parser.add_argument("--sample", help="Exact sample directory name.")
    parser.add_argument(
        "--filtered-root",
        type=Path,
        help="Optional override for the local-threshold input root.",
    )
    parser.add_argument("--level", type=int, default=0, help="Input OME-Zarr level.")
    parser.add_argument(
        "--view",
        choices=VIEWS,
        default="comparison",
        help=(
            "comparison: raw, normalized U-Net input, normalized U-Net result; "
            "all: add original-scale inputs, probability, and mask; "
            "result: saved result only."
        ),
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable napari grid mode and use ordinary layer toggling.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List completed sample names and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset = args.dataset or prompt_choice("Choose dataset", DATASETS)

    if args.output_root is not None:
        output_root = args.output_root.expanduser().resolve()
    else:
        roots = discover_inference_roots(dataset)
        if not roots:
            raise FileNotFoundError(
                f"No completed {dataset} inference run was found under "
                f"{PROJECT_ROOT / 'results' / 'U-net'}"
            )
        labels = [str(path.relative_to(PROJECT_ROOT)) for path in roots]
        selected_label = labels[0] if len(labels) == 1 else prompt_choice(
            "Choose inference run", labels
        )
        output_root = PROJECT_ROOT / selected_label

    samples = discover_samples(output_root)
    if args.list:
        print(f"\nCompleted samples under:\n{output_root}\n")
        for name in samples:
            print(name)
        return 0

    if args.sample is not None:
        sample = args.sample
        if sample not in samples:
            raise FileNotFoundError(
                f"Sample not found in inference output: {sample}\nAvailable:\n  "
                + "\n  ".join(samples)
            )
    else:
        sample = prompt_choice("Choose completed sample", list(samples))

    open_result(
        dataset=dataset,
        output_root=output_root,
        sample=sample,
        filtered_root_override=args.filtered_root,
        level=args.level,
        view=args.view,
        grid=not args.no_grid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
