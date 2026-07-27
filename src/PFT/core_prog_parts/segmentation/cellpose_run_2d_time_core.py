
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
    return max(0, int(n) // 2)


def _take_first_time(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = axes.lower()
    if 't' not in axes:
        return arr, axes
    ax = axes.index('t')
    return np.take(arr, 0, axis=ax), axes.replace('t', '')


def _move_yx_last(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = axes.lower()
    if axes.endswith('yx'):
        return arr, axes
    if 'y' not in axes or 'x' not in axes:
        raise ValueError(f"Cannot find y/x axes in axes={axes} shape={arr.shape}")
    y_i, x_i = axes.index('y'), axes.index('x')
    perm = [i for i in range(arr.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
    return np.transpose(arr, perm), ''.join(axes[i] for i in perm)


def _normalize_float32(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
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
    return project_root / 'results' / 'segmentation_finetuning_masks' / normalize_dataset_name(dataset)


def _model_dir(project_root: Path, dataset: str) -> Path:
    out = project_root / 'models' / f'cellpose_{normalize_dataset_name(dataset)}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prediction_root(project_root: Path, dataset: str) -> Path:
    out = project_root / 'results' / 'segmentation_cellpose' / normalize_dataset_name(dataset)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _candidate_image_paths(project_root: Path, dataset: str, sample: str) -> list[Path]:
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
    for p in _candidate_image_paths(project_root, dataset, sample):
        if p.exists():
            return p
    raise FileNotFoundError(f"No matching OME-Zarr image found for sample={sample!r}, dataset={dataset!r}")


def _list_input_images(project_root: Path, dataset: str) -> list[Path]:
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
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    arr = np.asarray(arr)
    axes = str(axes).lower()
    arr, axes = _take_first_time(arr, axes)
    return arr, axes


def _extract_2d_image(path: Path, *, channels: tuple[int, ...]) -> np.ndarray:
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
    m = np.asarray(tiff.imread(mask_path))
    if m.ndim == 3:
        m = m[_center_index(m.shape[0])]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={m.shape} from {mask_path}")
    return m.astype(np.int32, copy=False)


def _extract_3d_volume_as_slices(path: Path, *, channels: tuple[int, ...]) -> np.ndarray:
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
    m = np.asarray(tiff.imread(mask_path))
    if m.ndim == 4:
        # If mask accidentally has channel axis, keep first channel.
        m = m[0] if m.shape[0] <= 4 else m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask ZYX, got shape={m.shape} from {mask_path}")
    return m.astype(np.int32, copy=False)


def _collect_training_pairs(project_root: Path, dataset: str) -> list[tuple[Path, Path, str]]:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(np.uint16 if np.max(labels) <= np.iinfo(np.uint16).max else np.uint32)
    tiff.imwrite(out_dir / 'labels.tif', labels)
    zarr_path = out_dir / 'labels.ome.zarr'
    save_ome_zarr(zarr_path, labels, axes=axes, meta=None, overwrite=True, pyramid_3d=('z' in axes))
    return zarr_path



@dataclass
class CellposeRun2DTimeConfig:
    project_root: Path | None = None
    dataset: str = '2d_time'
    model_path: str | Path | None = None
    gpu: bool = True
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    batch_size: int = 8

    def root(self) -> Path:
        return self.project_root or find_project_root()


def run_cellpose_dataset(cfg: CellposeRun2DTimeConfig) -> Path:
    from cellpose import models

    project_root = cfg.root()
    model_path = Path(cfg.model_path) if cfg.model_path else _find_default_model(project_root, cfg.dataset)
    model = models.CellposeModel(gpu=cfg.gpu, pretrained_model=str(model_path))
    images = _list_input_images(project_root, cfg.dataset)

    out_root = _prediction_root(project_root, cfg.dataset) / model_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for zarr_path in images:
        sample = zarr_path.parent.name

        img = _extract_2d_image(zarr_path, channels=(0,))
        kwargs = dict(
            diameter=cfg.diameter,
            channel_axis=-1 if img.ndim == 3 else None,
            flow_threshold=cfg.flow_threshold,
            cellprob_threshold=cfg.cellprob_threshold,
            min_size=cfg.min_size,
            batch_size=cfg.batch_size,
            normalize=True,
        )
        masks, _flows, _styles = model.eval(img, **kwargs)
        labels = np.asarray(masks, dtype=np.int32)
        axes = 'yx'

        sample_dir = out_root / sample
        out_zarr = _save_labels_omezarr_and_tiff(sample_dir, labels, axes=axes)
        rows.append({
            'sample': sample,
            'input_omezarr': str(zarr_path),
            'model_path': str(model_path),
            'labels_omezarr': str(out_zarr),
            'labels_tif': str(sample_dir / 'labels.tif'),
            'labels_shape': list(labels.shape),
            'n_instances': int(len(np.unique(labels)) - (1 if np.any(labels == 0) else 0)),
        })

    (out_root / 'run_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return out_root
