from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, filtered_img_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

"""Train or fine-tune a 2D or 3D StarDist model on filtered OME-Zarr images and label masks."""


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
    """Normalize an image into a stable display and training range."""
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
    """Extract a 2D or 3D StarDist-ready image from one OME-Zarr file."""
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
            return _normalize_percentile(out_channels[0])
        return _normalize_percentile(np.stack(out_channels, axis=-1))

    if ndim == 3:
        out_channels = []
        for ch in channels:
            ch_axes = axes.replace("c", "")
            if "z" not in ch_axes:
                raise ValueError(f"3D mode requires a z axis in {path}, got axes={axes}")
            out_channels.append(np.asarray(ch, dtype=np.float32))
        if len(out_channels) == 1:
            return _normalize_percentile(out_channels[0])
        return _normalize_percentile(np.stack(out_channels, axis=-1))

    raise ValueError(f"Unsupported ndim={ndim}")


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


def load_training_data(project_root: Path, dataset: str, ndim: int):
    """Load normalized images, integer labels, and sample names for StarDist."""
    mask_root = _segmentation_masks_root(project_root, ndim, dataset)
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask folder does not exist: {mask_root}")

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    names: list[str] = []
    for sample_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        mask_path = next(iter(_mask_candidates(sample_dir)), None)
        if mask_path is None:
            continue
        try:
            img_path = _resolve_filtered_zarr(project_root, dataset, sample_dir.name)
        except FileNotFoundError:
            continue
        X.append(_extract_input_image(img_path, dataset, ndim))
        Y.append(_load_label_mask(mask_path, ndim))
        names.append(sample_dir.name)
    if not X:
        raise RuntimeError("No valid training pairs were found.")
    return X, Y, names


@dataclass
class StarDistTrainConfig:
    """Store training settings for one StarDist run."""
    project_root: Path
    dataset: str = "2d_time"
    ndim: int = 2
    model_name: str | None = None
    n_rays: int = 32
    grid: int = 2
    epochs: int = 200
    steps_per_epoch: int = 100
    train_batch_size: int = 4
    patch_size_2d: tuple[int, int] = (256, 256)
    patch_size_3d: tuple[int, int, int] = (32, 128, 128)
    learning_rate: float = 3e-4
    train_fraction: float = 0.8
    seed: int = 1337
    use_gpu: bool = True

    def run_dir(self) -> Path:
        """Build the output folder for one training run."""
        name = self.model_name or f"stardist_{self.ndim}d_{self.dataset}"
        out = self.project_root / "models" / name
        out.mkdir(parents=True, exist_ok=True)
        return out


def split_train_val(X, Y, names, train_fraction: float, seed: int):
    """Split loaded samples into training and validation sets."""
    idx = list(range(len(X)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = max(1, int(round(len(idx) * train_fraction)))
    train_idx = set(idx[:n_train])

    trX, trY, trN, vaX, vaY, vaN = [], [], [], [], [], []
    for i, (x, y, n) in enumerate(zip(X, Y, names)):
        if i in train_idx:
            trX.append(x)
            trY.append(y)
            trN.append(n)
        else:
            vaX.append(x)
            vaY.append(y)
            vaN.append(n)
    if not vaX:
        vaX, vaY, vaN = trX[-1:], trY[-1:], trN[-1:]
        trX, trY, trN = trX[:-1], trY[:-1], trN[:-1]
    return (trX, trY, trN), (vaX, vaY, vaN)


def train_stardist_model(cfg: StarDistTrainConfig) -> Path:
    """Train or fine-tune a 2D or 3D StarDist model and save it to disk."""
    X, Y, names = load_training_data(cfg.project_root, cfg.dataset, cfg.ndim)
    (train_X, train_Y, train_names), (val_X, val_Y, val_names) = split_train_val(
        X, Y, names, cfg.train_fraction, cfg.seed
    )

    if cfg.ndim == 2:
        from stardist.models import Config2D, StarDist2D

        n_channel_in = 1 if train_X[0].ndim == 2 else train_X[0].shape[-1]
        conf = Config2D(
            n_rays=cfg.n_rays,
            grid=(cfg.grid, cfg.grid),
            train_epochs=cfg.epochs,
            train_steps_per_epoch=cfg.steps_per_epoch,
            train_batch_size=cfg.train_batch_size,
            train_patch_size=cfg.patch_size_2d,
            train_learning_rate=cfg.learning_rate,
            n_channel_in=n_channel_in,
            use_gpu=cfg.use_gpu,
        )
        model = StarDist2D(conf, name=(cfg.model_name or f"stardist_2d_{cfg.dataset}"), basedir=str(cfg.project_root / "models"))
    else:
        from stardist.models import Config3D, StarDist3D

        n_channel_in = 1 if train_X[0].ndim == 3 else train_X[0].shape[-1]
        conf = Config3D(
            rays=cfg.n_rays,
            grid=(cfg.grid, cfg.grid, cfg.grid),
            train_epochs=cfg.epochs,
            train_steps_per_epoch=cfg.steps_per_epoch,
            train_batch_size=cfg.train_batch_size,
            train_patch_size=cfg.patch_size_3d,
            train_learning_rate=cfg.learning_rate,
            n_channel_in=n_channel_in,
            use_gpu=cfg.use_gpu,
        )
        model = StarDist3D(conf, name=(cfg.model_name or f"stardist_3d_{cfg.dataset}"), basedir=str(cfg.project_root / "models"))

    model.train(train_X, train_Y, validation_data=(val_X, val_Y))
    model.optimize_thresholds(val_X, val_Y)

    run_dir = cfg.run_dir()
    summary = {
        "dataset": cfg.dataset,
        "ndim": cfg.ndim,
        "epochs": cfg.epochs,
        "steps_per_epoch": cfg.steps_per_epoch,
        "train_batch_size": cfg.train_batch_size,
        "learning_rate": cfg.learning_rate,
        "n_rays": cfg.n_rays,
        "grid": cfg.grid,
        "train_samples": train_names,
        "val_samples": val_names,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def parse_args() -> StarDistTrainConfig:
    """Read command line settings for StarDist training."""
    project_root = find_project_root()
    p = argparse.ArgumentParser(description="Train or fine-tune StarDist on filtered OME-Zarr images.")
    p.add_argument("--dataset", default="2d_time")
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--model_name", default=None)
    p.add_argument("--n_rays", type=int, default=32)
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    return StarDistTrainConfig(
        project_root=project_root,
        dataset=args.dataset,
        ndim=args.ndim,
        model_name=args.model_name,
        n_rays=args.n_rays,
        grid=args.grid,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        train_fraction=args.train_fraction,
        seed=args.seed,
        use_gpu=not args.cpu,
    )


def main() -> None:
    """Run StarDist training from the terminal."""
    cfg = parse_args()
    out_dir = train_stardist_model(cfg)
    print(f"Saved StarDist training outputs to: {out_dir}")


if __name__ == "__main__":
    main()
