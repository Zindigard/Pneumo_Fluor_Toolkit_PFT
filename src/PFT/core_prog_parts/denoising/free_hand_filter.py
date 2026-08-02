"""Optional free-hand Fourier-mask filtering for 2D OME-Zarr images.

A free-hand filter multiplies the centered 2D Fourier transform by a user-
supplied keep mask. The method remains callable for comparison and manual
tuning, but it is not the production filter. The production 2D workflow uses
``apply_local_threshold_2d`` from ``local_threshold_filter.py``.
"""

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
    """Store a multiplicative Fourier-domain keep mask.

    Values must be finite and lie in [0, 1]. A value of one preserves a
    frequency coefficient; zero removes it; intermediate values attenuate it.
    """

    mask_keep: np.ndarray


def apply_freehand_filter_2d(img2d: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
    """Apply a supplied Fourier keep mask to one 2D image plane.

    Args:
        img2d (np.ndarray): Array containing img2d.
        mask_keep (np.ndarray): Array containing mask keep.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = apply_freehand_filter_2d(img2d=image_array, mask_keep=image_array)
    """
    x = np.asarray(img2d, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D image plane, received shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Free-hand filter input contains NaN or infinite values")
    mask_keep = np.asarray(mask_keep, dtype=np.float32)
    if not np.all(np.isfinite(mask_keep)):
        raise ValueError("mask_keep contains NaN or infinite values")
    if np.any(mask_keep < 0.0) or np.any(mask_keep > 1.0):
        raise ValueError("mask_keep values must be in [0, 1]")
    mu = float(np.mean(x))
    x0 = x - mu
    F = np.fft.fftshift(np.fft.fft2(x0))
    if mask_keep.shape != x.shape:
        raise ValueError(f"mask_keep shape {mask_keep.shape} != image shape {x.shape}")
    Ff = F * mask_keep.astype(np.float32, copy=False)
    y = np.fft.ifft2(np.fft.ifftshift(Ff))
    out = np.real(y).astype(np.float32) + mu
    return out


def _apply_mask_one_plane(img2d: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper for ``apply_freehand_filter_2d``.

    Args:
        img2d (np.ndarray): Array containing img2d.
        mask_keep (np.ndarray): Array containing mask keep.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _apply_mask_one_plane(img2d=image_array, mask_keep=image_array)
    """
    return apply_freehand_filter_2d(img2d, mask_keep)


def results_filters_dir() -> Path:
    """Return the common directory used for optional filter outputs.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = results_filters_dir()
    """
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
    """Apply an optional Fourier keep mask to one selected OME-Zarr image.

    Channel selection is dataset-aware: ``2d_time`` uses its blue channel and
    ``2d_wga_dapi`` can process blue and green. ``apply=False`` preserves the
    input values. The function returns the created output directory.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        params (FreehandMaskParams): Value specifying params for the operation.
        apply (bool): Boolean flag controlling apply.
        channel_mode (Literal["auto", "blue", "green"]): Value specifying channel mode for the operation. Defaults to ``"auto"``.
        image_index (int | None): Zero-based index selecting image. ``None`` selects the function's default behavior.
        out_subdir_name (str | None): Text value specifying out subdir name. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = run_freehand_on_dataset(
        ...     dataset="2d_time",
        ...     params=...,
        ...     apply=True,
        ... )
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
        """Internal helper used by this module.

        Returns:
            list[int]: Collection containing the generated or selected values.

        Example:
            >>> result = _pick_channels()
        """
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