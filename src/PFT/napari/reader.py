from __future__ import annotations
from pathlib import Path
from typing import Callable, Any
import numpy as np
from PFT.core_prog_parts.io import load_czi
from PFT.core_prog_parts.Ome_Zarr import save_ome_zarr_next_to_outputs 

LayerData = tuple[Any, dict, str]


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def sanitize_stem(name: str) -> str:
    """Windows-safe folder name (handles SIM² etc.)."""
    bad = r'<>:"/\|?*'
    return "".join(ch if ch not in bad else "_" for ch in name).strip()


def napari_zarr_dir_for_czi(czi_path: Path) -> Path:
    
    here = Path(__file__).resolve()
    root = find_project_root(here)

    base = root / "results" / "Napari_zarr"
    base.mkdir(parents=True, exist_ok=True)

    folder = base / sanitize_stem(czi_path.stem)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def pct_limits(img2d: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> tuple[float, float]:
    lo, hi = np.percentile(img2d, [p_low, p_high])
    lo = float(lo)
    hi = float(hi if hi > lo else lo + 1.0)
    return lo, hi


def napari_get_reader(path: str | list[str]) -> Callable[[str], list[LayerData]] | None:
    """Napari hook: return a reader callable if we support this path."""
    if isinstance(path, list):
        path = path[0]
    p = Path(path)

    if p.suffix.lower() == ".czi":
        return read_czi_autozarr
    return None


def read_czi_autozarr(path: str) -> list[LayerData]:
    """
    Drag/drop .czi:
      - load pixels (raw)
      - auto-export OME-Zarr into results/Napari_zarr/<file_stem>/
      - show image layers in napari (2D or 2-channel 2D)
      - attach the OME-Zarr path to layer.metadata["ome_zarr"]
    """
    p = Path(path)
    arr, meta = load_czi(p)
    arr = np.asarray(arr)

    out_dir = napari_zarr_dir_for_czi(p)

    # NOTE: if your function signature differs, adapt only this call.
    zarr_path = save_ome_zarr_next_to_outputs(
        ome_zarr_dir=out_dir,
        arr=arr,
        meta=meta,
        channel_names=getattr(meta, "channel_names", None),
    )

    # scaling 
    scale_yx = (meta.pixel_size_um_y, meta.pixel_size_um_x)

    layer_common = dict(
        scale=scale_yx,
        metadata={
            "source_czi": str(p),
            "ome_zarr": str(zarr_path),
            "czi_meta": meta,
        },
    )

    layers: list[LayerData] = []

    # 2D single channel (Y, X) ----
    if arr.ndim == 2:
        lo, hi = pct_limits(arr)
        kwargs = {
            **layer_common,
            "name": p.name,
            "contrast_limits": (lo, hi),
            "colormap": "blue",
        }
        layers.append((arr, kwargs, "image"))
        return layers

    #2D multi-channel (C, Y, X) or (Y, X, C)
    ch_names = getattr(meta, "channel_names", None)
    if not ch_names:
        if arr.ndim == 3:
            ch_names = [f"C{i}" for i in range(arr.shape[0] if arr.shape[0] <= 10 else arr.shape[-1])]
        else:
            ch_names = ["C0"]

    # channels first
    if arr.ndim == 3 and arr.shape[0] in (2, 3, 4):
        for c in range(arr.shape[0]):
            img = arr[c]
            lo, hi = pct_limits(img)
            cmap = "blue" if c == 0 else "green"
            kwargs = {
                **layer_common,
                "name": f"{p.stem} | {ch_names[c] if c < len(ch_names) else f'C{c}'}",
                "contrast_limits": (lo, hi),
                "colormap": cmap,
                "blending": "additive",
            }
            layers.append((img, kwargs, "image"))
        return layers

    # channels last
    if arr.ndim == 3 and arr.shape[-1] in (2, 3, 4):
        for c in range(arr.shape[-1]):
            img = arr[..., c]
            lo, hi = pct_limits(img)
            cmap = "blue" if c == 0 else "green"
            kwargs = {
                **layer_common,
                "name": f"{p.stem} | {ch_names[c] if c < len(ch_names) else f'C{c}'}",
                "contrast_limits": (lo, hi),
                "colormap": cmap,
                "blending": "additive",
            }
            layers.append((img, kwargs, "image"))
        return layers

    kwargs = {**layer_common, "name": f"{p.name} (raw)"}
    layers.append((arr, kwargs, "image"))
    return layers
