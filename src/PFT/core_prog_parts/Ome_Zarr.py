from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PFT.core_prog_parts.io import CziMeta
except Exception:  
    class CziMeta: 
        pass

_CANONICAL_TYPES = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}


def _axes_dicts(axes: str) -> list[dict]:
    """OME-NGFF axes as list-of-dicts."""
    out: list[dict] = []
    for a in axes:
        d = {"name": a, "type": _CANONICAL_TYPES.get(a, "unknown")}
        if a in "zyx":
            d["unit"] = "micrometer"
        out.append(d)
    return out


def _scale_vector_from_meta_um(axes: str, meta: CziMeta) -> list[float]:
    """Pixel size."""
    px_x = getattr(meta, "pixel_size_um_x", None)
    px_y = getattr(meta, "pixel_size_um_y", None)
    px_z = getattr(meta, "pixel_size_um_z", None)

    scale: list[float] = []
    for a in axes:
        if a == "x":
            scale.append(float(px_x) if px_x else 1.0)
        elif a == "y":
            scale.append(float(px_y) if px_y else 1.0)
        elif a == "z":
            scale.append(float(px_z) if px_z else 1.0)
        else:
            scale.append(1.0)
    return scale


def _infer_axes_for_squeezed(arr: np.ndarray, meta: CziMeta) -> str:
    """
    Infer an axes string for squeezed arrays.

    """
    if arr.ndim == 2:
        return "yx"

    if arr.ndim < 2:
        raise ValueError(f"Invalid array ndim={arr.ndim}")

    non_spatial = arr.ndim - 2

    ch_names = getattr(meta, "channel_names", None)
    n_ch = len(ch_names) if isinstance(ch_names, list) else None

    if n_ch is not None:
        for ax in range(non_spatial):
            if arr.shape[ax] == n_ch:
                if non_spatial == 1:
                    return "cyx"
                if non_spatial == 2:
                    # guess: CZYX (common for z-stacks) vs TCYX,if second dim looks like Z (not huge), choose CZYX
                    return "czyx" if arr.shape[1] <= 512 else "tcyx"
                if non_spatial == 3:
                    return "tczyx"

    if 2 <= arr.shape[0] <= 6:
        if non_spatial == 1:
            return "cyx"
        if non_spatial == 2:
            return "czyx"
        if non_spatial == 3:
            return "tczyx"

    # no channel 
    if non_spatial == 1:
        return "zyx"
    if non_spatial == 2:
        return "tzyx"

    raise ValueError(f"Cannot infer axes for shape={arr.shape}")


def save_ome_zarr(
    out_zarr_dir: str | Path,
    arr: np.ndarray,
    axes: str,
    meta: CziMeta | None = None,
    *,
    overwrite: bool = True,
    chunks: tuple[int, ...] | None = None,
) -> Path:
    """Write a single-scale OME-Zarr.

    Param
    ----------
    out_zarr_dir:
        Directory ending with `.ome.zarr` (it is a folder store).
    arr:
        Numpy array.
    axes:
        Axes string matching arr.ndim. Prefer subset of: t,c,z,y,x.
    meta:
        Optional CziMeta (adds pixel sizes + channel names into attrs).
    overwrite:
        If True, delete existing folder before writing.
    chunks:
        Optional zarr chunk shape.

    Returns
    -------
    Path to the written `.ome.zarr` folder.
    """
    out_zarr_dir = Path(out_zarr_dir)

    if len(axes) != arr.ndim:
        raise ValueError(f"axes '{axes}' does not match arr.ndim={arr.ndim} shape={arr.shape}")

    if overwrite and out_zarr_dir.exists():
        import shutil
        shutil.rmtree(out_zarr_dir)

    out_zarr_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.writer import write_image
    except Exception as e:
        raise ImportError(
            "OME-Zarr export requires `ome-zarr` + `zarr`.\n"
            "Install with: pip install ome-zarr zarr numcodecs"
        ) from e

    store = parse_url(str(out_zarr_dir), mode="w").store
    root = zarr.group(store=store)

    coord_tfs = None
    if meta is not None:
        coord_tfs = [[{"type": "scale", "scale": _scale_vector_from_meta_um(axes, meta)}]]

    write_image(
        image=arr,
        group=root,
        axes=_axes_dicts(axes),
        coordinate_transformations=coord_tfs,
        chunks=chunks,
        scaler=None, 
    )

    if meta is not None:
        # store extra attrs
        try:
            root.attrs["pft_meta"] = asdict(meta)  
        except Exception:
            root.attrs["pft_meta"] = {"source_path": getattr(meta, "source_path", "")}

        root.attrs["source_path"] = getattr(meta, "source_path", "")
        root.attrs["channel_names"] = getattr(meta, "channel_names", None)

    return out_zarr_dir


def save_ome_zarr_next_to_outputs(
    out_dir: str | Path,
    arr: np.ndarray,
    meta: CziMeta,
    *,
    overwrite: bool = True,
    chunks: tuple[int, ...] | None = None,
) -> Path:
    """Check folder for omezarr"""
    out_dir = Path(out_dir)
    out_zarr = out_dir / "image.ome.zarr"

    axes = _infer_axes_for_squeezed(arr, meta)

    # Ensure yx are last
    if not axes.endswith("yx") and ("y" in axes and "x" in axes):
        y_i, x_i = axes.index("y"), axes.index("x")
        perm = [i for i in range(arr.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
        arr = np.transpose(arr, perm)
        axes = "".join([axes[i] for i in perm])

    return save_ome_zarr(out_zarr, arr, axes, meta=meta, overwrite=overwrite, chunks=chunks)