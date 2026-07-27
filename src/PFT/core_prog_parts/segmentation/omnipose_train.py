from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, filtered_img_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

"""Prepare Omnipose training data from filtered OME-Zarr images and launch custom model training."""


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


def _load_filtered_image(path: Path) -> tuple[np.ndarray, str]:
    """Load one filtered OME-Zarr image as a numpy array."""
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    return np.asarray(arr), str(axes)


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


def _extract_input_image(path: Path, dataset: str, ndim: int) -> np.ndarray:
    """Extract a 2D or 3D Omnipose-ready image from one OME-Zarr file."""
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
            return _normalize_percentile(out_channels[0])[..., None]
        return _normalize_percentile(np.stack(out_channels, axis=-1))

    out_channels = []
    for ch in channels:
        ch_axes = axes.replace("c", "")
        if "z" not in ch_axes:
            raise ValueError(f"3D mode requires a z axis in {path}, got axes={axes}")
        out_channels.append(np.asarray(ch, dtype=np.float32))
    if len(out_channels) == 1:
        return _normalize_percentile(out_channels[0])[..., None]
    return _normalize_percentile(np.stack(out_channels, axis=-1))


def _mask_candidates(sample_dir: Path):
    """Yield likely mask files for one sample folder."""
    names = ["mask.tif", "mask.tiff", "masks.tif", "masks.tiff", "labels.tif", "labels.tiff"]
    for name in names:
        p = sample_dir / name
        if p.exists():
            yield p
    for ext in ("*.tif", "*.tiff"):
        for p in sorted(sample_dir.glob(ext)):
            yield p


def _load_label_mask(path: Path, ndim: int) -> np.ndarray:
    """Load one integer label mask from disk."""
    x = np.asarray(tiff.imread(path))
    if ndim == 2:
        if x.ndim == 3:
            x = x[_center_slice_index(x.shape[0])]
        if x.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {x.shape} from {path}")
    else:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape {x.shape} from {path}")
    return x.astype(np.int32)


def _resolve_filtered_zarr(project_root: Path, dataset: str, sample: str) -> Path:
    """Find the filtered OME-Zarr file for one sample."""
    ds = normalize_dataset_name(dataset)
    candidates = [
        filtered_img_root(project_root) / ds / sample / "image.ome.zarr",
        project_root / "results" / "img" / "filtered" / "2d" / ds / sample / "image.ome.zarr",
        project_root / "results" / "img" / "filtered" / "3d" / ds / sample / "image.ome.zarr",
        project_root / "results" / "img" / ds / sample / "image.ome.zarr",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Filtered OME-Zarr not found for sample '{sample}' in dataset '{ds}'")


def _segmentation_masks_root(project_root: Path, ndim: int, dataset: str) -> Path:
    """Return the segmentation mask folder for one dataset."""
    dim_dir = "2d" if ndim == 2 else "3d"
    return project_root / "results" / "segmentation_masks" / dim_dir / normalize_dataset_name(dataset)


def prepare_omnipose_training_folder(project_root: Path, dataset: str, ndim: int, out_dir: Path) -> Path:
    """Write Omnipose-style TIFF image and mask pairs into one folder."""
    mask_root = _segmentation_masks_root(project_root, ndim, dataset)
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask folder does not exist: {mask_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        mask_path = next(iter(_mask_candidates(sample_dir)), None)
        if mask_path is None:
            continue
        try:
            img_path = _resolve_filtered_zarr(project_root, dataset, sample_dir.name)
        except FileNotFoundError:
            continue

        img = _extract_input_image(img_path, dataset, ndim)
        mask = _load_label_mask(mask_path, ndim)
        tiff.imwrite(out_dir / f"{sample_dir.name}_img.tif", img.astype(np.float32))
        tiff.imwrite(out_dir / f"{sample_dir.name}_masks.tif", mask.astype(np.int32))
    return out_dir


@dataclass
class OmniposeTrainConfig:
    """Store training settings for one Omnipose run."""
    project_root: Path
    dataset: str = "2d_time"
    ndim: int = 2
    model_name: str = "omnipose_custom"
    pretrained_model: str = "None"
    nchan: int = 1
    nclasses: int = 3
    diameter: float = 0.0
    learning_rate: float = 0.1
    batch_size: int = 8
    n_epochs: int = 400
    use_gpu: bool = True
    extra_cli: list[str] | None = None

    def run_dir(self) -> Path:
        """Build the output folder for one training run."""
        out = self.project_root / "models" / self.model_name
        out.mkdir(parents=True, exist_ok=True)
        return out


def train_omnipose_model(cfg: OmniposeTrainConfig) -> Path:
    """Prepare Omnipose data and launch training through the official CLI."""
    run_dir = cfg.run_dir()
    data_dir = run_dir / "prepared_training_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    prepare_omnipose_training_folder(cfg.project_root, cfg.dataset, cfg.ndim, data_dir)

    cmd = [
        sys.executable,
        "-m",
        "omnipose",
        "--train",
        "--dir",
        str(data_dir),
        "--img_filter",
        "_img",
        "--mask_filter",
        "_masks",
        "--pretrained_model",
        str(cfg.pretrained_model),
        "--learning_rate",
        str(cfg.learning_rate),
        "--batch_size",
        str(cfg.batch_size),
        "--n_epochs",
        str(cfg.n_epochs),
        "--diameter",
        str(cfg.diameter),
        "--nchan",
        str(cfg.nchan),
        "--nclasses",
        str(cfg.nclasses),
        "--dim",
        str(cfg.ndim),
        "--save_every",
        "100",
    ]
    if cfg.nchan > 1:
        cmd.append("--all_channels")
    if cfg.use_gpu:
        cmd.append("--use_gpu")
    if cfg.extra_cli:
        cmd.extend(cfg.extra_cli)

    log_path = run_dir / "train_command.txt"
    log_path.write_text(" ".join(cmd), encoding="utf-8")
    subprocess.run(cmd, check=True, cwd=str(cfg.project_root))

    summary = {
        "dataset": cfg.dataset,
        "ndim": cfg.ndim,
        "model_name": cfg.model_name,
        "pretrained_model": cfg.pretrained_model,
        "nchan": cfg.nchan,
        "nclasses": cfg.nclasses,
        "diameter": cfg.diameter,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "n_epochs": cfg.n_epochs,
        "prepared_data_dir": str(data_dir),
        "command": cmd,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def parse_args() -> OmniposeTrainConfig:
    """Read command line settings for Omnipose training."""
    project_root = find_project_root()
    p = argparse.ArgumentParser(description="Train Omnipose on filtered OME-Zarr images.")
    p.add_argument("--dataset", default="2d_time")
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--model_name", default="omnipose_custom")
    p.add_argument("--pretrained_model", default="None")
    p.add_argument("--nchan", type=int, default=1)
    p.add_argument("--nclasses", type=int, default=3)
    p.add_argument("--diameter", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=400)
    p.add_argument("--cpu", action="store_true")
    args, extra = p.parse_known_args()
    return OmniposeTrainConfig(
        project_root=project_root,
        dataset=args.dataset,
        ndim=args.ndim,
        model_name=args.model_name,
        pretrained_model=args.pretrained_model,
        nchan=args.nchan,
        nclasses=args.nclasses,
        diameter=args.diameter,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        use_gpu=not args.cpu,
        extra_cli=extra,
    )


def main() -> None:
    """Run Omnipose training from the terminal."""
    cfg = parse_args()
    out_dir = train_omnipose_model(cfg)
    print(f"Saved Omnipose training outputs to: {out_dir}")


if __name__ == "__main__":
    main()
