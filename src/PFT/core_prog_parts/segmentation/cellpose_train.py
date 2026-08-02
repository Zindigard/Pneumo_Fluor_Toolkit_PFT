"""Provide command-line and programmatic utilities for cellpose train."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, filtered_img_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

"""Train or fine-tune a Cellpose model on filtered OME-Zarr images and instance masks."""


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
    x = np.take(x, indices=0, axis=t_idx)
    axes = axes.replace("t", "")
    return x, axes


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


def _load_filtered_image(path: Path) -> tuple[np.ndarray, str]:
    """Load one filtered OME-Zarr image as a numpy array.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Example:
        >>> result = _load_filtered_image(path=Path("path/to/resource"))
    """
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    return np.asarray(arr), str(axes)


def _extract_input_image(path: Path, dataset: str, ndim: int) -> np.ndarray:
    """Extract a model-ready 2D or 3D image from one OME-Zarr file.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        ndim (int): Numerical value controlling ndim.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _extract_input_image(
        ...     path=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     ndim=1,
        ... )
    """
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
            if ch.ndim != 2:
                raise ValueError(f"Expected 2D plane, got shape {ch.shape} from {path}")
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
            if ch.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {ch.shape} from {path}")
            out_channels.append(np.asarray(ch, dtype=np.float32))
        if len(out_channels) == 1:
            return out_channels[0]
        return np.stack(out_channels, axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported ndim={ndim}")


def _mask_candidates(sample_dir: Path) -> Iterable[Path]:
    """Yield likely mask files for one sample folder.

    Args:
        sample_dir (Path): Directory used for sample.

    Returns:
        Iterable[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _mask_candidates(sample_dir=Path("path/to/resource"))
    """
    names = [
        "mask.tif",
        "mask.tiff",
        "masks.tif",
        "masks.tiff",
        "labels.tif",
        "labels.tiff",
    ]
    for name in names:
        p = sample_dir / name
        if p.exists():
            yield p
    for ext in ("*.tif", "*.tiff"):
        for p in sorted(sample_dir.glob(ext)):
            yield p


def _load_label_mask(path: Path, ndim: int) -> np.ndarray:
    """Load one integer label mask from disk.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        ndim (int): Numerical value controlling ndim.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_label_mask(path=Path("path/to/resource"), ndim=1)
    """
    x = tiff.imread(path)
    x = np.asarray(x)
    if ndim == 2:
        if x.ndim == 3:
            x = x[_center_slice_index(x.shape[0])]
        if x.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {x.shape} from {path}")
    elif ndim == 3:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape {x.shape} from {path}")
    else:
        raise ValueError(f"Unsupported ndim={ndim}")
    return np.asarray(x, dtype=np.int32)


def _resolve_filtered_zarr(project_root: Path, dataset: str, sample: str) -> Path:
    """Find the filtered OME-Zarr file for one sample.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _resolve_filtered_zarr(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     sample="sample",
        ... )
    """
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
    """Return the segmentation mask folder for one dataset.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        ndim (int): Numerical value controlling ndim.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _segmentation_masks_root(
        ...     project_root=Path("path/to/resource"),
        ...     ndim=1,
        ...     dataset="2d_time",
        ... )
    """
    dim_dir = "2d" if ndim == 2 else "3d"
    return project_root / "results" / "segmentation_masks" / dim_dir / normalize_dataset_name(dataset)


def list_training_pairs(project_root: Path, dataset: str, ndim: int) -> list[tuple[Path, Path]]:
    """Collect filtered images and matching instance masks.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        ndim (int): Numerical value controlling ndim.

    Returns:
        list[tuple[Path, Path]]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = list_training_pairs(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     ndim=1,
        ... )
    """
    mask_root = _segmentation_masks_root(project_root, ndim, dataset)
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask folder does not exist: {mask_root}")

    pairs: list[tuple[Path, Path]] = []
    for sample_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        mask_path = next(iter(_mask_candidates(sample_dir)), None)
        if mask_path is None:
            continue
        try:
            img_path = _resolve_filtered_zarr(project_root, dataset, sample_dir.name)
        except FileNotFoundError:
            continue
        pairs.append((img_path, mask_path))
    return pairs


def load_training_data(project_root: Path, dataset: str, ndim: int) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load images, labels, and sample names for training.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        ndim (int): Numerical value controlling ndim.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[str]]: Collection containing the generated or selected values.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = load_training_data(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     ndim=1,
        ... )
    """
    pairs = list_training_pairs(project_root, dataset, ndim)
    if not pairs:
        raise RuntimeError("No valid training pairs were found.")

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for img_path, mask_path in pairs:
        images.append(_extract_input_image(img_path, dataset, ndim))
        labels.append(_load_label_mask(mask_path, ndim))
        names.append(mask_path.parent.name)
    return images, labels, names


@dataclass
class CellposeTrainConfig:
    """Store training settings for one Cellpose run."""
    project_root: Path
    dataset: str = "2d_time"
    ndim: int = 2
    pretrained_model: str = "cpsam"
    gpu: bool = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    batch_size: int = 4
    n_epochs: int = 200
    min_train_masks: int = 1
    test_fraction: float = 0.2
    seed: int = 1337
    normalize: bool = True
    compute_flows: bool = False
    save_every: int = 50
    model_name: str | None = None

    def run_dir(self) -> Path:
        """Build the output folder for one training run.

        Returns:
            Path: Resolved or generated filesystem path.

        Example:
            >>> instance = CellposeTrainConfig(...)
            >>> result = instance.run_dir()
        """
        name = self.model_name or f"cellpose_{self.dataset}_{self.ndim}d"
        out = self.project_root / "models" / name
        out.mkdir(parents=True, exist_ok=True)
        return out


def split_train_val(images: list[np.ndarray], labels: list[np.ndarray], names: list[str], test_fraction: float, seed: int):
    """Split loaded samples into training and validation sets.

    Args:
        images (list[np.ndarray]): Sequence or batch of input image arrays to process.
        labels (list[np.ndarray]): Integer label image in which each positive value identifies one segmented object.
        names (list[str]): Text value specifying names.
        test_fraction (float): Fractional value controlling test.
        seed (int): Random seed used to make sampling, splitting, or initialization reproducible.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = split_train_val(
        ...     images=image_array,
        ...     labels=image_array,
        ...     names="names",
        ...     test_fraction=0.5,
        ...     seed=1,
        ... )
    """
    idx = list(range(len(images)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_test = max(1, int(round(len(idx) * test_fraction))) if len(idx) > 1 else 0
    test_idx = set(idx[:n_test])

    tr_x, tr_y, tr_n = [], [], []
    va_x, va_y, va_n = [], [], []
    for i, (x, y, n) in enumerate(zip(images, labels, names)):
        if i in test_idx:
            va_x.append(x)
            va_y.append(y)
            va_n.append(n)
        else:
            tr_x.append(x)
            tr_y.append(y)
            tr_n.append(n)
    return (tr_x, tr_y, tr_n), (va_x, va_y, va_n)


def train_cellpose_model(cfg: CellposeTrainConfig) -> Path:
    """Train or fine-tune a Cellpose model and save it to disk.

    Args:
        cfg (CellposeTrainConfig): Value specifying cfg for the operation.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = train_cellpose_model(cfg=config)
    """
    from cellpose import models, train

    images, labels, names = load_training_data(cfg.project_root, cfg.dataset, cfg.ndim)
    (train_x, train_y, train_n), (val_x, val_y, val_n) = split_train_val(
        images, labels, names, cfg.test_fraction, cfg.seed
    )

    model = models.CellposeModel(gpu=cfg.gpu, pretrained_model=cfg.pretrained_model)
    run_dir = cfg.run_dir()
    model_name = cfg.model_name or f"cellpose_{cfg.dataset}_{cfg.ndim}d"

    kwargs = dict(
        train_data=train_x,
        train_labels=train_y,
        test_data=val_x if val_x else None,
        test_labels=val_y if val_y else None,
        channel_axis=-1 if (train_x[0].ndim in (3, 4)) else None,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.n_epochs,
        weight_decay=cfg.weight_decay,
        normalize=cfg.normalize,
        compute_flows=cfg.compute_flows,
        save_path=str(run_dir),
        save_every=cfg.save_every,
        min_train_masks=cfg.min_train_masks,
        model_name=model_name,
    )

    result = train.train_seg(model.net, **kwargs)

    summary = {
        "dataset": cfg.dataset,
        "ndim": cfg.ndim,
        "pretrained_model": cfg.pretrained_model,
        "gpu": cfg.gpu,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "n_epochs": cfg.n_epochs,
        "train_samples": train_n,
        "val_samples": val_n,
        "train_seg_result": str(result),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def parse_args() -> CellposeTrainConfig:
    """Read command line settings for Cellpose training.

    Returns:
        CellposeTrainConfig: Collection containing the generated or selected values.

    Example:
        >>> result = parse_args()
    """
    project_root = find_project_root()
    p = argparse.ArgumentParser(description="Train or fine-tune Cellpose on filtered OME-Zarr images.")
    p.add_argument("--dataset", default="2d_time")
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--pretrained_model", default="cpsam")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--min_train_masks", type=int, default=1)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--compute_flows", action="store_true")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--model_name", default=None)
    args = p.parse_args()

    gpu = True
    if args.cpu:
        gpu = False
    elif args.gpu:
        gpu = True

    return CellposeTrainConfig(
        project_root=project_root,
        dataset=args.dataset,
        ndim=args.ndim,
        pretrained_model=args.pretrained_model,
        gpu=gpu,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        min_train_masks=args.min_train_masks,
        test_fraction=args.test_fraction,
        seed=args.seed,
        normalize=not args.no_normalize,
        compute_flows=args.compute_flows,
        save_every=args.save_every,
        model_name=args.model_name,
    )


def main() -> None:
    """Run Cellpose fine-tuning from the terminal.

    Example:
        >>> exit_code = main()
    """
    cfg = parse_args()
    out_dir = train_cellpose_model(cfg)
    print(f"Saved Cellpose training outputs to: {out_dir}")


if __name__ == "__main__":
    main()
