from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from skimage.morphology import flood

from PFT.core_prog_parts.common_paths import find_project_root, ensure_dir

ArrayLike = np.ndarray


@dataclass(frozen=True)
class FloodFillResult:
    mask: np.ndarray
    seed_point: tuple[int, ...]
    seed_value: float
    tolerance: float | None
    filled_size: int
    ndim: int
    shape: tuple[int, ...]


def _validate_image(image: ArrayLike) -> np.ndarray:
    if image is None:
        raise TypeError("`image` must not be None.")

    arr = np.asarray(image)

    if arr.size == 0:
        raise ValueError("`image` must not be empty.")

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"`image` must be 2D or 3D, got shape {arr.shape} with ndim={arr.ndim}."
        )

    return arr


def _validate_seed_point(
    image: np.ndarray,
    seed_point: Sequence[int] | np.ndarray,
) -> tuple[int, ...]:
    if seed_point is None:
        raise ValueError("`seed_point` must not be None.")

    seed = tuple(int(v) for v in seed_point)

    if len(seed) != image.ndim:
        raise ValueError(
            f"`seed_point` must have {image.ndim} coordinates, got {len(seed)}."
        )

    for axis, (idx, size) in enumerate(zip(seed, image.shape)):
        if idx < 0 or idx >= size:
            raise ValueError(
                f"`seed_point` index out of bounds on axis {axis}: "
                f"{idx} not in [0, {size - 1}]."
            )

    return seed


def _validate_connectivity(connectivity: int, ndim: int) -> int:
    if not isinstance(connectivity, int):
        raise ValueError("`connectivity` must be an integer.")

    if connectivity < 1 or connectivity > ndim:
        raise ValueError(
            f"`connectivity` must be between 1 and image ndim ({ndim}), got {connectivity}."
        )

    return connectivity


def _resolve_tolerance(
    image: np.ndarray,
    tolerance: float | None = None,
    tolerance_rel: float | None = None,
) -> float | None:
    if tolerance is not None:
        tol = float(tolerance)
        if tol < 0:
            raise ValueError("`tolerance` must be >= 0.")
        return tol

    if tolerance_rel is None:
        return None

    tol_rel = float(tolerance_rel)
    if tol_rel < 0:
        raise ValueError("`tolerance_rel` must be >= 0.")

    image_min = float(np.min(image))
    image_max = float(np.max(image))
    dynamic_range = image_max - image_min

    if dynamic_range == 0:
        return 0.0

    return tol_rel * dynamic_range


def flood_mask(
    image: ArrayLike,
    seed_point: Sequence[int],
    tolerance: float | None = None,
    tolerance_rel: float | None = None,
    connectivity: int = 1,
) -> FloodFillResult:
    arr = _validate_image(image)
    seed = _validate_seed_point(arr, seed_point)
    conn = _validate_connectivity(connectivity, arr.ndim)
    tol = _resolve_tolerance(arr, tolerance=tolerance, tolerance_rel=tolerance_rel)

    mask = flood(arr, seed_point=seed, connectivity=conn, tolerance=tol)
    mask = np.asarray(mask, dtype=bool)

    return FloodFillResult(
        mask=mask,
        seed_point=seed,
        seed_value=float(arr[seed]),
        tolerance=tol,
        filled_size=int(mask.sum()),
        ndim=arr.ndim,
        shape=arr.shape,
    )


def flood_mask_from_percentile_tolerance(
    image: ArrayLike,
    seed_point: Sequence[int],
    percentile_fraction: float = 0.02,
    connectivity: int = 1,
) -> FloodFillResult:
    return flood_mask(
        image=image,
        seed_point=seed_point,
        tolerance=None,
        tolerance_rel=percentile_fraction,
        connectivity=connectivity,
    )


def merge_flood_masks(
    image: ArrayLike,
    seed_points: Iterable[Sequence[int]],
    tolerance: float | None = None,
    tolerance_rel: float | None = None,
    connectivity: int = 1,
) -> np.ndarray:
    arr = _validate_image(image)
    merged = np.zeros(arr.shape, dtype=bool)

    for seed in seed_points:
        result = flood_mask(
            image=arr,
            seed_point=seed,
            tolerance=tolerance,
            tolerance_rel=tolerance_rel,
            connectivity=connectivity,
        )
        merged |= result.mask

    return merged


def mask_to_uint8(mask: ArrayLike, true_value: int = 255) -> np.ndarray:
    if not (0 <= int(true_value) <= 255):
        raise ValueError("`true_value` must be in [0, 255].")

    mask_bool = np.asarray(mask, dtype=bool)
    out = np.zeros(mask_bool.shape, dtype=np.uint8)
    out[mask_bool] = np.uint8(true_value)
    return out


def apply_mask(
    image: ArrayLike,
    mask: ArrayLike,
    background_value: int | float = 0,
) -> np.ndarray:
    arr = np.asarray(image)
    m = np.asarray(mask, dtype=bool)

    if arr.shape != m.shape:
        raise ValueError(
            f"`image` and `mask` must have the same shape, got {arr.shape} and {m.shape}."
        )

    out = np.full(arr.shape, background_value, dtype=arr.dtype)
    out[m] = arr[m]
    return out


def default_floodfill_results_dir(project_root: Path | None = None) -> Path:
    """
    Optional helper for later runner scripts.
    Returns a standard project-relative folder for flood-fill outputs.
    """
    root = project_root or find_project_root(Path(__file__).resolve())
    return ensure_dir(root / "results" / "segmented" / "floodfill")


__all__ = [
    "FloodFillResult",
    "flood_mask",
    "flood_mask_from_percentile_tolerance",
    "merge_flood_masks",
    "mask_to_uint8",
    "apply_mask",
    "default_floodfill_results_dir",
]