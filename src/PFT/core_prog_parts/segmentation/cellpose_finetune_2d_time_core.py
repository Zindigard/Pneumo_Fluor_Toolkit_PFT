"""Provide command-line and programmatic utilities for cellpose finetune two-dimensional data time core."""


from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, ensure_czyx
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr


def _center_index(n: int) -> int:
    """Return center index for the supplied inputs.

    Args:
        n (int): Numerical value controlling n.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _center_index(n=1)
    """
    return max(0, int(n) // 2)


def _take_first_time(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Return take first time for the supplied inputs.

    Args:
        arr (np.ndarray): Array containing arr.
        axes (str): Axis specification describing the dimensional order of the image data.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Example:
        >>> result = _take_first_time(arr=image_array, axes="axes")
    """
    axes = axes.lower()
    if 't' not in axes:
        return arr, axes
    ax = axes.index('t')
    return np.take(arr, 0, axis=ax), axes.replace('t', '')


def _move_yx_last(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Return move yx last for the supplied inputs.

    Args:
        arr (np.ndarray): Array containing arr.
        axes (str): Axis specification describing the dimensional order of the image data.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _move_yx_last(arr=image_array, axes="axes")
    """
    axes = axes.lower()
    if axes.endswith('yx'):
        return arr, axes
    if 'y' not in axes or 'x' not in axes:
        raise ValueError(f"Cannot find y/x axes in axes={axes} shape={arr.shape}")
    y_i, x_i = axes.index('y'), axes.index('x')
    perm = [i for i in range(arr.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
    return np.transpose(arr, perm), ''.join(axes[i] for i in perm)


def _normalize_float32(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Normalize float32 using the configured procedure.

    Args:
        img (np.ndarray): Array containing img.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _normalize_float32(img=image_array)
    """
    x = np.asarray(img, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _mask_candidates(sample_dir: Path) -> Iterable[Path]:
    """Return mask candidates for the supplied inputs.

    Args:
        sample_dir (Path): Directory used for sample.

    Returns:
        Iterable[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _mask_candidates(sample_dir=Path("path/to/resource"))
    """
    names = [
        'mask.tif', 'mask.tiff', 'masks.tif', 'masks.tiff',
        'label.tif', 'label.tiff', 'labels.tif', 'labels.tiff',
        'instance_mask.tif', 'instance_mask.tiff',
    ]
    for name in names:
        p = sample_dir / name
        if p.exists():
            yield p
    for ext in ('*.tif', '*.tiff'):
        for p in sorted(sample_dir.glob(ext)):
            yield p


def _finetune_mask_root(project_root: Path, dataset: str) -> Path:
    """Return finetune mask root for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _finetune_mask_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    return project_root / 'results' / 'segmentation_finetuning_masks' / normalize_dataset_name(dataset)


def _model_dir(project_root: Path, dataset: str) -> Path:
    """Return model dir for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _model_dir(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    out = project_root / 'models' / f'cellpose_{normalize_dataset_name(dataset)}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prediction_root(project_root: Path, dataset: str) -> Path:
    """Return prediction root for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _prediction_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    out = project_root / 'results' / 'segmentation_cellpose' / normalize_dataset_name(dataset)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _candidate_image_paths(project_root: Path, dataset: str, sample: str) -> list[Path]:
    """Return candidate image paths for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _candidate_image_paths(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     sample="sample",
        ... )
    """
    ds = normalize_dataset_name(dataset)
    return [
        project_root / 'results' / 'denoised_images' / ds / sample / 'image.ome.zarr',
        project_root / 'results' / 'denoised_images' / ds / sample / 'pred_mask.ome.zarr',
        project_root / 'results' / 'denoised_images' / '2d' / ds / sample / 'image.ome.zarr',
        project_root / 'results' / 'denoised_images' / '3d' / sample / 'image.ome.zarr',
        project_root / 'results' / 'denoised_images' / '3d' / sample / 'pred_mask.ome.zarr',
        project_root / 'results' / 'img' / 'filtered' / ds / sample / 'image.ome.zarr',
        project_root / 'results' / 'img' / '2d_data' / ds / sample / 'image.ome.zarr',
        project_root / 'results' / 'img' / '3d_data' / sample / 'image.ome.zarr',
    ]


def _resolve_image(project_root: Path, dataset: str, sample: str) -> Path:
    """Resolve image from the supplied configuration.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _resolve_image(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     sample="sample",
        ... )
    """
    for p in _candidate_image_paths(project_root, dataset, sample):
        if p.exists():
            return p
    raise FileNotFoundError(f"No matching OME-Zarr image found for sample={sample!r}, dataset={dataset!r}")


def _list_input_images(project_root: Path, dataset: str) -> list[Path]:
    """List input images available in the configured project structure.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _list_input_images(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    ds = normalize_dataset_name(dataset)
    roots = [
        project_root / 'results' / 'denoised_images' / ds,
        project_root / 'results' / 'denoised_images' / '2d' / ds,
        project_root / 'results' / 'denoised_images' / '3d',
        project_root / 'results' / 'img' / 'filtered' / ds,
        project_root / 'results' / 'img' / '2d_data' / ds,
        project_root / 'results' / 'img' / '3d_data',
    ]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for name in ('image.ome.zarr', 'pred_mask.ome.zarr'):
                z = sample_dir / name
                if z.exists():
                    out.append(z)
                    break
    unique = sorted({str(p.resolve()): p for p in out}.values())
    if not unique:
        raise FileNotFoundError(f"No OME-Zarr inputs found for dataset={dataset!r}")
    return unique


def _load_ome(path: Path) -> tuple[np.ndarray, str]:
    """Load ome from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Example:
        >>> result = _load_ome(path=Path("path/to/resource"))
    """
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    arr = np.asarray(arr)
    axes = str(axes).lower()
    arr, axes = _take_first_time(arr, axes)
    return arr, axes


def _extract_2d_image(path: Path, *, channels: tuple[int, ...]) -> np.ndarray:
    """Extract two-dimensional data image from the supplied data.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        channels (tuple[int, ...]): Channel indices or identifiers selected for processing.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _extract_2d_image(path=Path("path/to/resource"), channels=1)
    """
    arr, axes = _load_ome(path)
    arr, axes = _move_yx_last(arr, axes)
    if 'z' in axes:
        z_ax = axes.index('z')
        arr = np.take(arr, _center_index(arr.shape[z_ax]), axis=z_ax)
        axes = axes.replace('z', '')
    if 'c' in axes:
        c_ax = axes.index('c')
        planes = []
        for ch in channels:
            if ch < arr.shape[c_ax]:
                plane = np.take(arr, ch, axis=c_ax)
                planes.append(_normalize_float32(plane))
        if not planes:
            raise ValueError(f"No requested channels {channels} found in {path}, shape={arr.shape}, axes={axes}")
        if len(planes) == 1:
            return planes[0]
        return np.stack(planes, axis=-1).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after extraction, got shape={arr.shape}, axes={axes}, path={path}")
    return _normalize_float32(arr)


def _load_2d_mask(mask_path: Path) -> np.ndarray:
    """Load two-dimensional data mask from persistent storage.

    Args:
        mask_path (Path): Filesystem path associated with mask.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_2d_mask(mask_path=Path("path/to/resource"))
    """
    m = np.asarray(tiff.imread(mask_path))
    if m.ndim == 3:
        m = m[_center_index(m.shape[0])]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={m.shape} from {mask_path}")
    return m.astype(np.int32, copy=False)


def _extract_3d_volume_as_slices(path: Path, *, channels: tuple[int, ...]) -> np.ndarray:
    """Extract three-dimensional data volume as slices from the supplied data.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        channels (tuple[int, ...]): Channel indices or identifiers selected for processing.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _extract_3d_volume_as_slices(path=Path("path/to/resource"), channels=1)
    """
    arr, axes = _load_ome(path)
    arr, axes = ensure_czyx(arr, axes)
    arr = np.asarray(arr)
    planes = []
    for ch in channels:
        if ch < arr.shape[0]:
            planes.append(arr[ch])  # ZYX
    if not planes:
        raise ValueError(f"No requested channels {channels} found in {path}, shape={arr.shape}")
    norm = [_normalize_float32(v) for v in planes]
    if len(norm) == 1:
        return norm[0].astype(np.float32)       # ZYX
    return np.stack(norm, axis=-1).astype(np.float32)  # ZYXC


def _load_3d_mask(mask_path: Path) -> np.ndarray:
    """Load three-dimensional data mask from persistent storage.

    Args:
        mask_path (Path): Filesystem path associated with mask.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_3d_mask(mask_path=Path("path/to/resource"))
    """
    m = np.asarray(tiff.imread(mask_path))
    if m.ndim == 4:
        # If mask accidentally has channel axis, keep first channel.
        m = m[0] if m.shape[0] <= 4 else m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask ZYX, got shape={m.shape} from {mask_path}")
    return m.astype(np.int32, copy=False)


def _collect_training_pairs(project_root: Path, dataset: str) -> list[tuple[Path, Path, str]]:
    """Collect training pairs from the available inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[tuple[Path, Path, str]]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _collect_training_pairs(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    root = _finetune_mask_root(project_root, dataset)
    if not root.exists():
        raise FileNotFoundError(f"Fine-tuning mask folder not found: {root}")
    pairs: list[tuple[Path, Path, str]] = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mask = next(iter(_mask_candidates(sample_dir)), None)
        if mask is None:
            continue
        try:
            img = _resolve_image(project_root, dataset, sample_dir.name)
        except FileNotFoundError:
            continue
        pairs.append((img, mask, sample_dir.name))
    if not pairs:
        raise RuntimeError(f"No valid Cellpose fine-tuning pairs found in {root}")
    return pairs


def _find_default_model(project_root: Path, dataset: str) -> Path:
    """Find default model in the available data or project structure.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _find_default_model(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    root = _model_dir(project_root, dataset)
    candidates = []
    for pattern in ('**/*.pth', '**/*cellpose*', '**/*_best*', '**/*'):
        for p in root.glob(pattern):
            if p.is_file() and p.stat().st_size > 0:
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No trained Cellpose model found under {root}. Train first or pass model_path.")
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _save_labels_omezarr_and_tiff(out_dir: Path, labels: np.ndarray, axes: str) -> Path:
    """Save labels OME-Zarr and TIFF data to persistent storage.

    Args:
        out_dir (Path): Directory used for out.
        labels (np.ndarray): Integer label image in which each positive value identifies one segmented object.
        axes (str): Axis specification describing the dimensional order of the image data.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _save_labels_omezarr_and_tiff(
        ...     out_dir=Path("path/to/resource"),
        ...     labels=image_array,
        ...     axes="axes",
        ... )
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(np.uint16 if np.max(labels) <= np.iinfo(np.uint16).max else np.uint32)
    tiff.imwrite(out_dir / 'labels.tif', labels)
    zarr_path = out_dir / 'labels.ome.zarr'
    save_ome_zarr(zarr_path, labels, axes=axes, meta=None, overwrite=True, pyramid_3d=('z' in axes))
    return zarr_path



@dataclass
class CellposeFineTune2DTimeConfig:
    """Store validated configuration or result data for cellpose fine tune2 dtime config."""
    project_root: Path | None = None
    dataset: str = '2d_time'
    pretrained_model: str = 'cpsam'
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
    model_name: str = 'cellpose_2d_time'

    def root(self) -> Path:
        """Return root for the supplied inputs.

        Returns:
            Path: Resolved or generated filesystem path.

        Example:
            >>> instance = CellposeFineTune2DTimeConfig(...)
            >>> result = instance.root()
        """
        return self.project_root or find_project_root()

    def run_dir(self) -> Path:
        """Run dir using the supplied configuration.

        Returns:
            Path: Resolved or generated filesystem path.

        Example:
            >>> instance = CellposeFineTune2DTimeConfig(...)
            >>> result = instance.run_dir()
        """
        return _model_dir(self.root(), self.dataset)


def _split_train_val(images: list[np.ndarray], labels: list[np.ndarray], names: list[str], fraction: float, seed: int):
    """Split train val into the requested subsets.

    Args:
        images (list[np.ndarray]): Sequence or batch of input image arrays to process.
        labels (list[np.ndarray]): Integer label image in which each positive value identifies one segmented object.
        names (list[str]): Text value specifying names.
        fraction (float): Numerical value controlling fraction.
        seed (int): Random seed used to make sampling, splitting, or initialization reproducible.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = _split_train_val(
        ...     images=image_array,
        ...     labels=image_array,
        ...     names="names",
        ...     fraction=0.5,
        ...     seed=1,
        ... )
    """
    idx = list(range(len(images)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(idx) * fraction))) if len(idx) > 1 else 0
    val_idx = set(idx[:n_val])
    tr_x, tr_y, tr_n, va_x, va_y, va_n = [], [], [], [], [], []
    for i, (x, y, n) in enumerate(zip(images, labels, names)):
        if i in val_idx:
            va_x.append(x); va_y.append(y); va_n.append(n)
        else:
            tr_x.append(x); tr_y.append(y); tr_n.append(n)
    return (tr_x, tr_y, tr_n), (va_x, va_y, va_n)


def finetune_cellpose(cfg: CellposeFineTune2DTimeConfig) -> Path:
    """Return finetune cellpose for the supplied inputs.

    Args:
        cfg (CellposeFineTune2DTimeConfig): Value specifying cfg for the operation.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = finetune_cellpose(cfg=config)
    """
    from cellpose import models, train

    project_root = cfg.root()
    pairs = _collect_training_pairs(project_root, cfg.dataset)

    train_images, train_labels, sample_names = [], [], []
    for img_path, mask_path, sample in pairs:
        train_images.append(_extract_2d_image(img_path, channels=(0,)))
        train_labels.append(_load_2d_mask(mask_path))
        sample_names.append(sample)

    (tr_x, tr_y, tr_n), (va_x, va_y, va_n) = _split_train_val(
        train_images, train_labels, sample_names, cfg.test_fraction, cfg.seed
    )
    if not tr_x:
        raise RuntimeError('No training samples after split. Add more masks or reduce test_fraction.')

    run_dir = cfg.run_dir()
    model = models.CellposeModel(gpu=cfg.gpu, pretrained_model=cfg.pretrained_model)
    result = train.train_seg(
        model.net,
        train_data=tr_x,
        train_labels=tr_y,
        test_data=va_x if va_x else None,
        test_labels=va_y if va_y else None,
        channel_axis=-1 if train_images[0].ndim == 3 else None,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.n_epochs,
        weight_decay=cfg.weight_decay,
        normalize=cfg.normalize,
        compute_flows=cfg.compute_flows,
        save_path=str(run_dir),
        save_every=cfg.save_every,
        min_train_masks=cfg.min_train_masks,
        model_name=cfg.model_name,
    )

    summary = asdict(cfg)
    summary.update({
        'project_root': str(project_root),
        'run_dir': str(run_dir),
        'input_dataset': cfg.dataset,
        'image_channels': (0,),
        'mode': "2d",
        'n_train_arrays': len(tr_x),
        'n_val_arrays': len(va_x),
        'train_samples': tr_n[:200],
        'val_samples': va_n[:200],
        'cellpose_train_result': str(result),
    })
    (run_dir / 'train_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return run_dir
