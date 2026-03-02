from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
try:
    from PFT.core_prog_parts.io import CziMeta
except Exception:
    class CziMeta:  # fallback type
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
    """Pixel size scale vector from meta (µm)."""
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
                    # CZYX (common for z-stacks) vs TCYX
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


def _should_make_pyramid_for_axes(axes: str) -> bool:
    """
    We only want pyramid for 3D volumes (i.e., data has a Z axis).
    Examples:
      - 'czyx'  -> True
      - 'tczyx' -> True
      - 'yx'    -> False
      - 'cyx'   -> False
    """
    axes = (axes or "").lower()
    return "z" in axes


def _scale_vector_for_level_um(
    axes: str,
    base_scale_um: list[float],
    *,
    level: int,
    downscale: int,
) -> list[float]:
   
    s = []
    factor = (downscale ** level)
    for a, v in zip(axes, base_scale_um):
        if a in ("y", "x"):
            s.append(float(v) * factor)
        else:
            s.append(float(v))
    return s


def save_ome_zarr(
    out_zarr_dir: str | Path,
    arr: np.ndarray,
    axes: str,
    meta: CziMeta | None = None,
    *,
    overwrite: bool = True,
    chunks: tuple[int, ...] | None = None,
    pyramid_3d: bool = True,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
) -> Path:
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
        from ome_zarr.scale import Scaler
    except Exception as e:
        raise ImportError(
            "OME-Zarr export requires `ome-zarr` + `zarr`.\n"
            "Install with: pip install ome-zarr zarr numcodecs"
        ) from e

    store = parse_url(str(out_zarr_dir), mode="w").store
    root = zarr.group(store=store)

    use_pyramid = bool(pyramid_3d) and ("z" in (axes or "").lower()) and pyramid_max_layer >= 1

    coord_tfs = None
    if meta is not None:
        base_scale_um = _scale_vector_from_meta_um(axes, meta)

        if use_pyramid:
            coord_tfs = []
            for lvl in range(pyramid_max_layer + 1):
                lvl_scale = _scale_vector_for_level_um(
                    axes, base_scale_um, level=lvl, downscale=pyramid_downscale
                )
                coord_tfs.append([{"type": "scale", "scale": lvl_scale}])
        else:
            coord_tfs = [[{"type": "scale", "scale": base_scale_um}]]

    scaler = None
    if use_pyramid:
        scaler = Scaler(downscale=pyramid_downscale, max_layer=pyramid_max_layer)

    write_image(
        image=arr,
        group=root,
        axes=_axes_dicts(axes),
        coordinate_transformations=coord_tfs,
        chunks=chunks,
        scaler=scaler,
    )

    if meta is not None:
        try:
            root.attrs["pft_meta"] = asdict(meta)
        except Exception:
            root.attrs["pft_meta"] = {"source_path": getattr(meta, "source_path", "")}

        root.attrs["source_path"] = getattr(meta, "source_path", "")
        root.attrs["channel_names"] = getattr(meta, "channel_names", None)

    root.attrs["pft_multiscale_enabled"] = bool(use_pyramid and scaler is not None)
    root.attrs["pft_multiscale_reason"] = "3d_pyramid" if (use_pyramid and scaler is not None) else "single_scale"

    return out_zarr_dir


def save_ome_zarr_next_to_outputs(
    out_dir: str | Path,
    arr: np.ndarray,
    meta: CziMeta,
    *,
    overwrite: bool = True,
    chunks: tuple[int, ...] | None = None,
    pyramid_3d: bool = True,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
) -> Path:
    """Write image.ome.zarr next to an output directory."""
    out_dir = Path(out_dir)
    out_zarr = out_dir / "image.ome.zarr"

    axes = _infer_axes_for_squeezed(arr, meta)

    # Ensure yx are last
    if not axes.endswith("yx") and ("y" in axes and "x" in axes):
        y_i, x_i = axes.index("y"), axes.index("x")
        perm = [i for i in range(arr.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
        arr = np.transpose(arr, perm)
        axes = "".join([axes[i] for i in perm])

    return save_ome_zarr(
        out_zarr,
        arr,
        axes,
        meta=meta,
        overwrite=overwrite,
        chunks=chunks,
        pyramid_3d=pyramid_3d,
        pyramid_max_layer=pyramid_max_layer,
        pyramid_downscale=pyramid_downscale,
    )