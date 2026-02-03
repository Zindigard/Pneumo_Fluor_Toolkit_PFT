from __future__ import annotations

"""Decode OME-Zarr into arrays that are convenient for Noise2Void (N2V).

This module is intentionally *read/convert only*.

This decoder provides:
- `load_ome_zarr(...)` -> (array, axes)
- `ome_zarr_to_n2v_2d_stack(...)` -> numpy array shaped (N, Y, X, 1)

Notes
-----
- OME-Zarr can be lazy (Dask).Should be implemented. 
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np


def load_ome_zarr(
    zarr_dir: str | Path,
    *,
    level: int = 0,
    as_numpy: bool = False,
) -> Tuple[Any, str]:
    """
    Load OME-Zarr and return (array, axes).
    """
    zarr_dir = Path(zarr_dir)

    try:
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader
    except Exception as e:
        raise ImportError(
            "Reading OME-Zarr requires `ome-zarr` (and usually `dask`).\n"
            "Install with: pip install ome-zarr zarr numcodecs dask"
        ) from e

    loc = parse_url(str(zarr_dir), mode="r")
    reader = Reader(loc)
    nodes = list(reader())
    if not nodes:
        raise ValueError(f"No image nodes found in: {zarr_dir}")

    node = nodes[0]
    data = node.data[level]

    axes: str | None = None
    try:
        ms = node.metadata.get("multiscales", [])
        if ms and "axes" in ms[0]:
            axes_entries = ms[0]["axes"]
            axes = "".join(a["name"] if isinstance(a, dict) else str(a) for a in axes_entries)
    except Exception:
        axes = None

    if axes is None:
        ndim = getattr(data, "ndim", None)
        if ndim == 2:
            axes = "yx"
        elif ndim == 3:
            axes = "cyx"
        elif ndim == 4:
            axes = "tcyx"
        elif ndim == 5:
            axes = "tczyx"
        else:
            axes = "unknown"

    if as_numpy:
        data = np.asarray(data)

    return data, axes


def ome_zarr_to_n2v_2d_stack(
    zarr_dir: str | Path,
    *,
    channel: int = 0,
    time: int | None = None,
    z: int | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Convert OME-Zarr into a NumPy stack shaped (N, Y, X, 1) for 2D N2V.

    - If 'c' exists -> pick `channel`
    - If `time` is not None and 't' exists -> pick that time index
    - If `z` is not None and 'z' exists -> pick that z index

    Returns
    -------
    x: np.ndarray
        Shape (N, Y, X, 1), dtype float32 if normalized else original dtype promoted.
    """
    data, axes = load_ome_zarr(zarr_dir, as_numpy=False)

    arr = data
    axes_str = axes

    def take_axis(letter: str, idx: int) -> None:
        nonlocal arr, axes_str
        if letter not in axes_str:
            return
        ax = axes_str.index(letter)
        arr = arr.take(indices=idx, axis=ax)
        axes_str = axes_str[:ax] + axes_str[ax + 1 :]

    take_axis("c", channel)
    if time is not None:
        take_axis("t", time)
    if z is not None:
        take_axis("z", z)

    # yx last
    if axes_str != "yx" and ("y" in axes_str and "x" in axes_str):
        y_i, x_i = axes_str.index("y"), axes_str.index("x")
        perm = [i for i in range(len(axes_str)) if i not in (y_i, x_i)] + [y_i, x_i]
        arr = arr.transpose(perm)
        axes_str = "".join(axes_str[i] for i in perm)

    arr_np = np.asarray(arr)

    if axes_str == "yx":
        stack = arr_np[None, ...]
    elif axes_str.endswith("yx"):
        n = int(np.prod(arr_np.shape[:-2]))
        y, x = arr_np.shape[-2], arr_np.shape[-1]
        stack = arr_np.reshape(n, y, x)
    else:
        raise ValueError(f"Cannot convert to 2D N2V stack from axes='{axes_str}' shape={arr_np.shape}")

    #  normalization
    if normalize == "percentile":
        stack = stack.astype(np.float32, copy=False)
        out = np.empty_like(stack, dtype=np.float32)
        for i in range(stack.shape[0]):
            frame = stack[i]
            lo = np.percentile(frame, 1.0)
            hi = np.percentile(frame, 99.8)
            if hi <= lo:
                out[i] = 0.0
            else:
                out[i] = (frame - lo) / (hi - lo)
                out[i] = np.clip(out[i], 0.0, 1.0)
        stack = out
    elif normalize is not None:
        raise ValueError("normalize must be None or 'percentile'")

    return stack[..., None]