from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, filtered_img_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

"""Run an Omnipose model on filtered OME-Zarr images and save label masks."""


def _take_first_time(x: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Remove time by taking the first frame when present."""
    if "t" not in axes:
        return x, axes
    t_idx = axes.index("t")
    return np.take(x, indices=0, axis=t_idx), axes.replace("t", "")


def _center_slice_index(length: int) -> int:
    """Return the center index for one dimension."""
    return max(0, int(length) // 2)


def _pick_channels(dataset: str, n_available: int) -> list[int]:
    """Choose the image channels used for one dataset."""
    ds = normalize_dataset_name(dataset)
    if ds == "2d_wga_dapi":
        return list(range(min(2, n_available)))
    return [0]


def _normalize_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.8) -> np.ndarray:
    """Normalize image intensities into a stable range."""
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
    """Extract a 2D or 3D Omnipose-ready image from one OME-Zarr file."""
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
    """List filtered OME-Zarr files for one dataset."""
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
class OmniposeRunConfig:
    """Store prediction settings for one Omnipose run."""
    project_root: Path
    dataset: str = "2d_time"
    ndim: int = 2
    model_name: str | None = None
    pretrained_model: str | None = None
    model_type: str = "cyto2_omni"
    nchan: int = 1
    nclasses: int = 3
    diameter: float | None = None
    mask_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_size: int = 15
    gpu: bool = True

    def output_root(self) -> Path:
        """Build the prediction output folder."""
        name = self.model_name or self.model_type
        out = self.project_root / "results" / "segmentation_predictions" / "omnipose" / self.dataset / name
        out.mkdir(parents=True, exist_ok=True)
        return out


def build_model(cfg: OmniposeRunConfig):
    """Load a built-in or custom Omnipose model."""
    from cellpose_omni import models

    if cfg.pretrained_model:
        try:
            return models.CellposeModel(
                gpu=cfg.gpu,
                pretrained_model=cfg.pretrained_model,
                nchan=cfg.nchan,
                nclasses=cfg.nclasses,
                dim=cfg.ndim,
            )
        except TypeError:
            return models.CellposeModel(gpu=cfg.gpu, pretrained_model=cfg.pretrained_model)

    try:
        return models.Cellpose(
            gpu=cfg.gpu,
            model_type=cfg.model_type,
            nchan=cfg.nchan,
            nclasses=cfg.nclasses,
            dim=cfg.ndim,
        )
    except TypeError:
        return models.Cellpose(gpu=cfg.gpu, model_type=cfg.model_type)


def run_one_image(model, img: np.ndarray, cfg: OmniposeRunConfig) -> np.ndarray:
    """Run Omnipose on one image or volume."""
    channels = [0, 0] if cfg.nchan == 1 else [2, 1]
    kwargs = dict(
        diameter=cfg.diameter,
        channels=channels,
        omni=True,
        mask_threshold=cfg.mask_threshold,
        flow_threshold=cfg.flow_threshold,
        min_size=cfg.min_size,
    )
    if cfg.ndim == 3:
        kwargs["do_3D"] = True
    masks, _flows, _styles, _diams = model.eval([img], **kwargs)
    return np.asarray(masks[0], dtype=np.int32)


def run_omnipose_dataset(cfg: OmniposeRunConfig) -> Path:
    """Run Omnipose on all filtered images of one dataset."""
    model = build_model(cfg)
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


def parse_args() -> OmniposeRunConfig:
    """Read command line settings for Omnipose prediction."""
    project_root = find_project_root()
    p = argparse.ArgumentParser(description="Run Omnipose on filtered OME-Zarr images.")
    p.add_argument("--dataset", default="2d_time")
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--model_name", default=None)
    p.add_argument("--pretrained_model", default=None)
    p.add_argument("--model_type", default="cyto2_omni")
    p.add_argument("--nchan", type=int, default=1)
    p.add_argument("--nclasses", type=int, default=3)
    p.add_argument("--diameter", type=float, default=None)
    p.add_argument("--mask_threshold", type=float, default=0.0)
    p.add_argument("--flow_threshold", type=float, default=0.4)
    p.add_argument("--min_size", type=int, default=15)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    gpu = True
    if args.cpu:
        gpu = False
    elif args.gpu:
        gpu = True

    return OmniposeRunConfig(
        project_root=project_root,
        dataset=args.dataset,
        ndim=args.ndim,
        model_name=args.model_name,
        pretrained_model=args.pretrained_model,
        model_type=args.model_type,
        nchan=args.nchan,
        nclasses=args.nclasses,
        diameter=args.diameter,
        mask_threshold=args.mask_threshold,
        flow_threshold=args.flow_threshold,
        min_size=args.min_size,
        gpu=gpu,
    )


def main() -> None:
    """Run Omnipose prediction from the terminal."""
    cfg = parse_args()
    out_dir = run_omnipose_dataset(cfg)
    print(f"Saved Omnipose predictions to: {out_dir}")


if __name__ == "__main__":
    main()
