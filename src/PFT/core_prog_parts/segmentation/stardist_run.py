"""Provide command-line and programmatic utilities for stardist run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, filtered_img_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

"""Run a 2D or 3D StarDist model on filtered OME-Zarr images and save label masks."""


def _take_first_time(x: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Remove time by taking the first frame when present.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        axes (str): Axis specification describing the dimensional order of the image data.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Example:
        >>> result = _take_first_time(x=image_array, axes="axes")
    """
    if "t" not in axes:
        return x, axes
    t_idx = axes.index("t")
    return np.take(x, indices=0, axis=t_idx), axes.replace("t", "")


def _center_slice_index(length: int) -> int:
    """Return the center index for one dimension.

    Args:
        length (int): Numerical value controlling length.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _center_slice_index(length=1)
    """
    return max(0, int(length) // 2)


def _pick_channels(dataset: str, n_available: int) -> list[int]:
    """Choose the image channels used for one dataset.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        n_available (int): Number of available used by the operation.

    Returns:
        list[int]: Collection containing the generated or selected values.

    Example:
        >>> result = _pick_channels(dataset="2d_time", n_available=1)
    """
    ds = normalize_dataset_name(dataset)
    if ds == "2d_wga_dapi":
        return list(range(min(2, n_available)))
    return [0]


def _normalize_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.8) -> np.ndarray:
    """Normalize an image into a stable prediction range.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        p_lo (float): Numerical value controlling p lo. Defaults to ``1.0``.
        p_hi (float): Numerical value controlling p hi. Defaults to ``99.8``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _normalize_percentile(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        lo = np.percentile(x, p_lo)
        hi = np.percentile(x, p_hi)
        return np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)

    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        xc = x[..., c]
        lo = np.percentile(xc, p_lo)
        hi = np.percentile(xc, p_hi)
        out[..., c] = np.clip((xc - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return out.astype(np.float32)


def extract_input_image(path: Path, dataset: str, ndim: int) -> np.ndarray:
    """Extract a 2D or 3D StarDist-ready image from one OME-Zarr file.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        ndim (int): Numerical value controlling ndim.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = extract_input_image(
        ...     path=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     ndim=1,
        ... )
    """
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    x = np.asarray(arr)
    axes = str(axes)
    x, axes = _take_first_time(x, axes)

    c_idx = axes.index("c") if "c" in axes else None
    if c_idx is None:
        channels = [x]
    else:
        ch_ids = _pick_channels(dataset, x.shape[c_idx])
        channels = [np.take(x, indices=c, axis=c_idx) for c in ch_ids]

    if ndim == 2:
        out_channels: list[np.ndarray] = []
        for ch in channels:
            ch_axes = axes.replace("c", "")
            if "z" in ch_axes:
                z_idx = ch_axes.index("z")
                ch = np.take(ch, indices=_center_slice_index(ch.shape[z_idx]), axis=z_idx)
            out_channels.append(np.asarray(ch, dtype=np.float32))
        if len(out_channels) == 1:
            return _normalize_percentile(out_channels[0])
        return _normalize_percentile(np.stack(out_channels, axis=-1))

    out_channels = []
    for ch in channels:
        ch_axes = axes.replace("c", "")
        if "z" not in ch_axes:
            raise ValueError(f"3D mode requires a z axis in {path}, got axes={axes}")
        out_channels.append(np.asarray(ch, dtype=np.float32))
    if len(out_channels) == 1:
        return _normalize_percentile(out_channels[0])
    return _normalize_percentile(np.stack(out_channels, axis=-1))


def list_filtered_images(project_root: Path, dataset: str) -> list[Path]:
    """List filtered OME-Zarr files for one dataset.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = list_filtered_images(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    ds = normalize_dataset_name(dataset)
    candidates = [
        filtered_img_root(project_root) / ds,
        project_root / "results" / "img" / "filtered" / "2d" / ds,
        project_root / "results" / "img" / "filtered" / "3d" / ds,
    ]
    roots = [p for p in candidates if p.exists()]
    if not roots:
        raise FileNotFoundError(f"No filtered dataset folder found for {ds}")

    out: list[Path] = []
    for root in roots:
        for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            zarr = sample_dir / "image.ome.zarr"
            if zarr.exists():
                out.append(zarr)
    unique = sorted({p.resolve() for p in out})
    return [Path(p) for p in unique]


@dataclass
class StarDistRunConfig:
    """Store prediction settings for one StarDist run."""
    project_root: Path
    dataset: str = "2d_time"
    ndim: int = 2
    model_name: str = "stardist_2d_2d_time"
    prob_thresh: float | None = None
    nms_thresh: float | None = None

    def output_root(self) -> Path:
        """Build the prediction output folder.

        Returns:
            Path: Resolved or generated filesystem path.

        Example:
            >>> instance = StarDistRunConfig(...)
            >>> result = instance.output_root()
        """
        out = self.project_root / "results" / "segmentation_predictions" / "stardist" / self.dataset / self.model_name
        out.mkdir(parents=True, exist_ok=True)
        return out


def load_model(cfg: StarDistRunConfig):
    """Load a 2D or 3D StarDist model from disk.

    Args:
        cfg (StarDistRunConfig): Value specifying cfg for the operation.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = load_model(cfg=config)
    """
    basedir = cfg.project_root / "models"
    if cfg.ndim == 2:
        from stardist.models import StarDist2D
        return StarDist2D(None, name=cfg.model_name, basedir=str(basedir))
    from stardist.models import StarDist3D
    return StarDist3D(None, name=cfg.model_name, basedir=str(basedir))


def run_one_image(model, img: np.ndarray, cfg: StarDistRunConfig) -> np.ndarray:
    """Run StarDist on one image or volume.

    Args:
        model (Any): Model identifier or filesystem path to the pretrained or fine-tuned model.
        img (np.ndarray): Array containing img.
        cfg (StarDistRunConfig): Value specifying cfg for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = run_one_image(
        ...     model="model_name",
        ...     img=image_array,
        ...     cfg=config,
        ... )
    """
    kwargs = {}
    if cfg.prob_thresh is not None:
        kwargs["prob_thresh"] = cfg.prob_thresh
    if cfg.nms_thresh is not None:
        kwargs["nms_thresh"] = cfg.nms_thresh
    labels, _details = model.predict_instances(img, **kwargs)
    return np.asarray(labels, dtype=np.int32)


def run_stardist_dataset(cfg: StarDistRunConfig) -> Path:
    """Run StarDist on all filtered images of one dataset.

    Args:
        cfg (StarDistRunConfig): Value specifying cfg for the operation.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = run_stardist_dataset(cfg=config)
    """
    model = load_model(cfg)
    images = list_filtered_images(cfg.project_root, cfg.dataset)
    out_root = cfg.output_root()

    rows: list[dict] = []
    for zarr_path in images:
        sample = zarr_path.parent.name
        img = extract_input_image(zarr_path, cfg.dataset, cfg.ndim)
        labels = run_one_image(model, img, cfg)
        sample_dir = out_root / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(sample_dir / "labels.tif", labels.astype(np.int32))
        rows.append({
            "sample": sample,
            "image_path": str(zarr_path),
            "labels_path": str(sample_dir / "labels.tif"),
            "shape": list(labels.shape),
            "n_instances": int(len(np.unique(labels)) - (1 if np.any(labels == 0) else 0)),
        })

    (out_root / "run_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_root


def parse_args() -> StarDistRunConfig:
    """Read command line settings for StarDist prediction.

    Returns:
        StarDistRunConfig: Generated or resolved text value.

    Example:
        >>> result = parse_args()
    """
    project_root = find_project_root()
    p = argparse.ArgumentParser(description="Run StarDist on filtered OME-Zarr images.")
    p.add_argument("--dataset", default="2d_time")
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--model_name", required=True)
    p.add_argument("--prob_thresh", type=float, default=None)
    p.add_argument("--nms_thresh", type=float, default=None)
    args = p.parse_args()
    return StarDistRunConfig(
        project_root=project_root,
        dataset=args.dataset,
        ndim=args.ndim,
        model_name=args.model_name,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
    )


def main() -> None:
    """Run StarDist prediction from the terminal.

    Example:
        >>> exit_code = main()
    """
    cfg = parse_args()
    out_dir = run_stardist_dataset(cfg)
    print(f"Saved StarDist predictions to: {out_dir}")


if __name__ == "__main__":
    main()
