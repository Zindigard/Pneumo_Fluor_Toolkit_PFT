from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.denoising.notch_filter import (
    _repo_root,
    _dataset_dir,
    list_omezarr_images,
    _to_numpy,
    _ensure_cyx,
)


@dataclass
class FreehandMaskParams:
    """
    Free-hand filter = user-defined frequency mask applied to FFT.
    
    """
    mask_keep: np.ndarray


def _apply_mask_one_plane(img2d: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
    """Internal helper used by this module."""
    x = img2d.astype(np.float32, copy=False)
    mu = float(np.mean(x))
    x0 = x - mu
    F = np.fft.fftshift(np.fft.fft2(x0))
    if mask_keep.shape != x.shape:
        raise ValueError(f"mask_keep shape {mask_keep.shape} != image shape {x.shape}")
    Ff = F * mask_keep.astype(np.float32, copy=False)
    y = np.fft.ifft2(np.fft.ifftshift(Ff))
    out = np.real(y).astype(np.float32) + mu
    return out


def results_filters_dir() -> Path:
    """Helper function used by this module."""
    return _repo_root() / "results" / "Filters"


def run_freehand_on_dataset(
    dataset: str,
    params: FreehandMaskParams,
    *,
    apply: bool,
    channel_mode: Literal["auto", "blue", "green"] = "auto",
    image_index: int | None = None,
    out_subdir_name: str | None = None,
) -> Path:
    """
    Apply freehand mask (or dry-run) to ONE selected image from dataset,
    save result as OME-Zarr 

    Returns output directory.
    """
    zarrs = list_omezarr_images(dataset)
    if not zarrs:
        raise FileNotFoundError(
            f"No OME-Zarr images found for dataset={dataset}. "
            f"Looked in: {_dataset_dir(dataset)}"
        )

    if image_index is None:
        image_index = 0
    image_index = int(np.clip(image_index, 0, len(zarrs) - 1))
    in_path = zarrs[image_index]
    stem = in_path.parent.name

    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)

    if "c" in axes:
        c_i = axes.index("c")
        n_c = x.shape[c_i]
    else:
        n_c = 1

    def _pick_channels() -> list[int]:
        """Internal helper used by this module."""
        if channel_mode == "blue":
            return [0]
        if channel_mode == "green":
            return [1] if n_c > 1 else [0]
        # auto:
        if dataset.strip().lower() == "2d_time":
            return [0]
        return list(range(min(n_c, 2))) if n_c > 1 else [0]

    chs = _pick_channels()

    y = x.astype(np.float32, copy=True)

    if apply:
        for c in chs:
            sl = [slice(None)] * y.ndim
            if "c" in axes:
                sl[axes.index("c")] = c
            plane = y[tuple(sl)]
            if plane.ndim == 2:
                y[tuple(sl)] = _apply_mask_one_plane(plane, params.mask_keep)
            elif plane.ndim == 3 and "t" in axes:
                for t in range(plane.shape[0]):
                    plane[t] = _apply_mask_one_plane(plane[t], params.mask_keep)
                y[tuple(sl)] = plane
            else:
                raise ValueError(f"Unexpected plane ndim={plane.ndim} for axes={axes}")
    else:
        pass  # dry-run

    out_root = results_filters_dir() / "Free_hand" / dataset
    if out_subdir_name:
        out_root = out_root / out_subdir_name
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = SimpleNamespace(
        pixel_size_um_x=1.0,
        pixel_size_um_y=1.0,
        pixel_size_um_z=1.0,
        channel_names=["ch0", "ch1", "ch2"][: n_c],
        source_path=str(in_path),
        axes=axes,
    )

    save_ome_zarr_next_to_outputs(
        out_dir,
        y,
        meta,
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
    )

    return out_dir