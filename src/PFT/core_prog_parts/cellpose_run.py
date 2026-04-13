from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, filtered_img_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

"""Run a Cellpose model on filtered OME-Zarr images and save label masks."""


def _take_first_time(x: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Remove time by taking the first frame when present."""
    if "t" not in axes:
        return x, axes
    t_idx = axes.index("t")
    return np.take(x, indices=0, axis=t_idx), axes.replace("t", "")


def _pick_channels(dataset: str, n_available: int) -> list[int]:
    """Choose the image channels used for one dataset."""
    ds = normalize_dataset_name(dataset)
    if ds == "2d_wga_dapi":
        return list(range(min(2, n_available)))
    return [0]


def _center_slice_index(length: int) -> int:
    """Return the center index for one dimension."""
    return max(0, int(length) // 2)


def _load_filtered_image(path: Path) -> tuple[np.ndarray, str]:
    """Load one filtered OME-Zarr image as a numpy array."""
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    return np.asarray(arr), str(axes)


def extract_input_image(path: Path, dataset: str, ndim: int) -> np.ndarray:
    """Extract a model-ready 2D or 3D image from one OME-Zarr file."""
    x, axes = _load_filtered_image(path)
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
            return out_channels[0]
        return np.stack(out_channels, axis=-1).astype(np.float32)

    if ndim == 3:
        out_channels = []
        for ch in channels:
            ch_axes = axes.replace("c", "")
            if "z" not in ch_axes:
                raise ValueError(f"3D mode requires a z axis in {path}, got axes={axes}")
            out_channels.append(np.asarray(ch, dtype=np.float32))
        if len(out_channels) == 1:
            return out_channels[0]
        return np.stack(out_channels, axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported ndim={ndim}")


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
class CellposeRunConfig:
    """Store prediction settings for one Cellpose run."""
    project_root: Path
    dataset: str = "2d_time"
    ndim: int = 2
    pretrained_model: str = "cpsam"
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    batch_size: int = 8
    gpu: bool = True
    model_name: str | None = None

    def output_root(self) -> Path:
        """Build the prediction output folder."""
        name = self.model_name or Path(self.pretrained_model).stem
        out = self.project_root / "results" / "segmentation_predictions" / "cellpose" / self.dataset / name
        out.mkdir(parents=True, exist_ok=True)
        return out


def run_one_image(model, img: np.ndarray, cfg: CellposeRunConfig) -> np.ndarray:
    """Run Cellpose on one image or volume."""
    kwargs = dict(
        batch_size=cfg.batch_size,
        flow_threshold=cfg.flow_threshold,
        cellprob_threshold=cfg.cellprob_threshold,
        min_size=cfg.min_size,
        diameter=cfg.diameter,
        normalize=True,
    )
    if cfg.ndim == 3:
        kwargs.update(dict(do_3D=True, z_axis=0, channel_axis=-1 if img.ndim == 4 else None))
    else:
        kwargs.update(dict(channel_axis=-1 if img.ndim == 3 else None))

    masks, _flows, _styles = model.eval(img, **kwargs)
    return np.asarray(masks, dtype=np.int32)


def run_cellpose_dataset(cfg: CellposeRunConfig) -> Path:
    """Run Cellpose on all filtered images of one dataset."""
    from cellpose import models

    model = models.CellposeModel(gpu=cfg.gpu, pretrained_model=cfg.pretrained_model)
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


def parse_args() -> CellposeRunConfig:
    """Read command line settings for Cellpose prediction."""
    project_root = find_project_root()
    p = argparse.ArgumentParser(description="Run Cellpose on filtered OME-Zarr images.")
    p.add_argument("--dataset", default="2d_time")
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--pretrained_model", default="cpsam")
    p.add_argument("--diameter", type=float, default=None)
    p.add_argument("--flow_threshold", type=float, default=0.4)
    p.add_argument("--cellprob_threshold", type=float, default=0.0)
    p.add_argument("--min_size", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--model_name", default=None)
    args = p.parse_args()

    gpu = True
    if args.cpu:
        gpu = False
    elif args.gpu:
        gpu = True

    return CellposeRunConfig(
        project_root=project_root,
        dataset=args.dataset,
        ndim=args.ndim,
        pretrained_model=args.pretrained_model,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        batch_size=args.batch_size,
        gpu=gpu,
        model_name=args.model_name,
    )


def main() -> None:
    """Run Cellpose prediction from the terminal."""
    cfg = parse_args()
    out_dir = run_cellpose_dataset(cfg)
    print(f"Saved Cellpose predictions to: {out_dir}")


if __name__ == "__main__":
    main()
