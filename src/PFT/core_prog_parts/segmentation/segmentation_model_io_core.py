from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, ensure_czyx
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr

DATASETS_2D = ("2d_time", "2d_wga_dapi")
DATASET_3D = "3d"


def center_index(n: int) -> int:
    return max(0, int(n) // 2)


def safe_sample_name(path: Path) -> str:
    p = Path(path)
    if p.name.endswith(".zarr") and p.parent.name.endswith(".ome"):
        return p.parent.parent.name
    if p.name == "image.ome.zarr" or p.name == "pred_mask.ome.zarr":
        return p.parent.name
    return p.stem


def normalize_float32(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    x = normalize_float32(img)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def take_first_time(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = str(axes).lower()
    if "t" not in axes:
        return arr, axes
    t_ax = axes.index("t")
    return np.take(arr, 0, axis=t_ax), axes.replace("t", "")


def move_yx_last(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = str(axes).lower()
    if axes.endswith("yx"):
        return arr, axes
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"Cannot find y/x axes in axes={axes}, shape={arr.shape}")
    y_i, x_i = axes.index("y"), axes.index("x")
    perm = [i for i in range(arr.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
    return np.transpose(arr, perm), "".join(axes[i] for i in perm)


def load_ome(path: Path, level: int = 0) -> tuple[np.ndarray, str]:
    arr, axes = load_ome_zarr(Path(path), level=level, as_numpy=False)
    arr = np.asarray(arr)
    axes = str(axes).lower()
    arr, axes = take_first_time(arr, axes)
    return arr, axes


def dataset_channels(dataset: str) -> tuple[int, ...]:
    ds = normalize_dataset_name(dataset)
    if ds == "2d_wga_dapi":
        return (0, 1)
    return (0,)


def extract_2d_image(path: Path, dataset: str, channels: Sequence[int] | None = None) -> np.ndarray:
    arr, axes = load_ome(path)
    arr, axes = move_yx_last(arr, axes)
    if "z" in axes:
        z_ax = axes.index("z")
        arr = np.take(arr, center_index(arr.shape[z_ax]), axis=z_ax)
        axes = axes.replace("z", "")
    channels = tuple(dataset_channels(dataset) if channels is None else channels)
    if "c" in axes:
        c_ax = axes.index("c")
        planes = []
        for ch in channels:
            if ch < arr.shape[c_ax]:
                planes.append(normalize_float32(np.take(arr, ch, axis=c_ax)))
        if not planes:
            raise ValueError(f"No selected channels {channels} in {path}, shape={arr.shape}, axes={axes}")
        if len(planes) == 1:
            return planes[0]
        return np.stack(planes, axis=-1).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after extraction, got shape={arr.shape}, axes={axes}, path={path}")
    return normalize_float32(arr)


def extract_3d_volume(path: Path, channels: Sequence[int] | None = None) -> np.ndarray:
    arr, axes = load_ome(path)
    arr, axes = ensure_czyx(arr, axes)
    arr = np.asarray(arr)
    channels = tuple((0,) if channels is None else channels)
    planes = []
    for ch in channels:
        if ch < arr.shape[0]:
            planes.append(normalize_float32(arr[ch]))
    if not planes:
        raise ValueError(f"No selected channels {channels} in {path}, shape={arr.shape}")
    if len(planes) == 1:
        return planes[0].astype(np.float32)       # ZYX
    return np.stack(planes, axis=-1).astype(np.float32)  # ZYXC


def mask_candidates(sample_dir: Path) -> Iterable[Path]:
    names = [
        "mask.tif", "mask.tiff", "masks.tif", "masks.tiff",
        "label.tif", "label.tiff", "labels.tif", "labels.tiff",
        "instance_mask.tif", "instance_mask.tiff",
    ]
    for name in names:
        p = sample_dir / name
        if p.exists():
            yield p
    for ext in ("*.tif", "*.tiff"):
        for p in sorted(sample_dir.glob(ext)):
            yield p


def load_2d_mask(mask_path: Path) -> np.ndarray:
    m = np.asarray(tiff.imread(mask_path))
    if m.ndim == 3:
        m = m[center_index(m.shape[0])]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={m.shape} from {mask_path}")
    return m.astype(np.int32, copy=False)


def load_3d_mask(mask_path: Path) -> np.ndarray:
    m = np.asarray(tiff.imread(mask_path))
    if m.ndim == 4:
        m = m[0] if m.shape[0] <= 4 else m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask ZYX, got shape={m.shape} from {mask_path}")
    return m.astype(np.int32, copy=False)


def finetune_mask_root(project_root: Path, dataset: str) -> Path:
    return Path(project_root) / "results" / "segmentation_finetuning_masks" / normalize_dataset_name(dataset)


def candidate_image_paths(project_root: Path, dataset: str, sample: str) -> list[Path]:
    root = Path(project_root)
    ds = normalize_dataset_name(dataset)
    if ds == "3d":
        return [
            root / "results" / "denoised_images" / "3d" / sample / "image.ome.zarr",
            root / "results" / "denoised_images" / "3d" / sample / "pred_mask.ome.zarr",
            root / "results" / "img" / "3d_data" / sample / "image.ome.zarr",
        ]
    return [
        root / "results" / "denoised_images" / ds / sample / "image.ome.zarr",
        root / "results" / "denoised_images" / ds / sample / "pred_mask.ome.zarr",
        root / "results" / "denoised_images" / "2d" / ds / sample / "image.ome.zarr",
        root / "results" / "denoised_images" / "2d" / ds / sample / "pred_mask.ome.zarr",
        root / "results" / "img" / "filtered" / ds / sample / "image.ome.zarr",
        root / "results" / "img" / "2d_data" / ds / sample / "image.ome.zarr",
    ]


def resolve_image(project_root: Path, dataset: str, sample: str) -> Path:
    for p in candidate_image_paths(project_root, dataset, sample):
        if p.exists():
            return p
    raise FileNotFoundError(f"No matching OME-Zarr image found for sample={sample!r}, dataset={dataset!r}")


def list_input_images(project_root: Path, dataset: str) -> list[Path]:
    root = Path(project_root)
    ds = normalize_dataset_name(dataset)
    if ds == "3d":
        roots = [
            root / "results" / "denoised_images" / "3d",
            root / "results" / "img" / "3d_data",
        ]
    else:
        roots = [
            root / "results" / "denoised_images" / ds,
            root / "results" / "denoised_images" / "2d" / ds,
            root / "results" / "img" / "filtered" / ds,
            root / "results" / "img" / "2d_data" / ds,
        ]
    out: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for sample_dir in sorted(p for p in r.iterdir() if p.is_dir()):
            for name in ("image.ome.zarr", "pred_mask.ome.zarr"):
                z = sample_dir / name
                if z.exists():
                    out.append(z)
                    break
    unique = sorted({str(p.resolve()): p for p in out}.values())
    if not unique:
        raise FileNotFoundError(f"No OME-Zarr inputs found for dataset={dataset!r}")
    return unique


def collect_training_pairs(project_root: Path, dataset: str) -> list[tuple[Path, Path, str]]:
    root = finetune_mask_root(project_root, dataset)
    if not root.exists():
        raise FileNotFoundError(f"Fine-tuning mask folder not found: {root}")
    pairs = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mask = next(iter(mask_candidates(sample_dir)), None)
        if mask is None:
            continue
        try:
            img = resolve_image(project_root, dataset, sample_dir.name)
        except FileNotFoundError:
            continue
        pairs.append((img, mask, sample_dir.name))
    if not pairs:
        raise RuntimeError(f"No valid training pairs found in {root}")
    return pairs


def save_label_outputs(labels: np.ndarray, out_dir: Path, axes: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(np.uint16, copy=False)
    tiff.imwrite(out_dir / "labels.tif", labels)
    zarr_path = out_dir / "labels.ome.zarr"
    save_ome_zarr(labels, zarr_path, axes=axes)
    return zarr_path


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    def default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")


def model_root(project_root: Path, family: str, dataset: str) -> Path:
    out = Path(project_root) / "models" / f"{family}_{normalize_dataset_name(dataset)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def prediction_root(project_root: Path, family: str, dataset: str, model_name: str | None = None) -> Path:
    name = model_name or "default_model"
    out = Path(project_root) / "results" / f"segmentation_{family}" / normalize_dataset_name(dataset) / name
    out.mkdir(parents=True, exist_ok=True)
    return out
