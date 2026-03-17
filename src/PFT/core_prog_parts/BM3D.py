from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import _to_numpy, _ensure_cyx, list_omezarr_images, results_filters_dir
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs  


@dataclass
class BM3DParams:
    """
    BM3D parameters you can tune (most important first):

    1) sigma_psd:
       - Noise std estimate in the *normalized* [0..1] intensity domain.
       - Typical starting values:
         * light denoise: 0.02 - 0.06
         * medium:        0.06 - 0.12
         * strong:        0.12 - 0.20
       - If you OVER-estimate sigma -> over-smoothing.
       - If you UNDER-estimate sigma -> little effect.

    2) stage:
       - BM3DStages.HARD_THRESHOLDING: faster, weaker
       - BM3DStages.ALL_STAGES: best quality, slower (recommended)

    3) profile:
       - 'np' is usually fine. Some bm3d packages expose profiles like 'ref', 'fast', etc.
       - If available in your bm3d version, use it to trade quality vs speed.

    4) post_gaussian_sigma:
       - Optional extra smoothing after BM3D (like your Denoising.py used ndimage.gaussian_filter).
       - 0.0 disables it. Typical: 0.0 - 0.8

    5) blend_original:
       - Optional blending of denoised result with original (helps preserve edges).
       - 0.0 -> fully original, 1.0 -> fully denoised.
       - Typical: 0.7 - 1.0

    6) normalize_mode:
       - 'dtype': normalize using dtype max (uint16->65535)
       - 'minmax': normalize per-image using min/max (can be unstable across images)
       - 'percentile': robust normalization per-image (recommended for display, sometimes for denoise)
         but if you want consistent sigma across dataset, prefer 'dtype'.

    7) preserve_range:
       - If True, output is rescaled back to original input range (based on normalization choice).
       - If False, output stays in [0..1] float32.

    Notes:
    - BM3D expects float images; if your images are uint16 raw, we denoise in float32.
    """
    sigma_psd: float = 0.08
    stage: Literal["all", "hard"] = "all"
    profile: str | None = None
    post_gaussian_sigma: float = 0.0
    blend_original: float = 1.0
    normalize_mode: Literal["dtype", "minmax", "percentile"] = "dtype"
    percentile_lo: float = 1.0
    percentile_hi: float = 99.5
    preserve_range: bool = True


def _require_bm3d():
    """Internal helper used by this module."""
    try:
        import bm3d  # type: ignore
        return bm3d
    except Exception as e:
        raise ImportError(
            "BM3D requires the `bm3d` package. Install with: pip install bm3d"
        ) from e


def _normalize_to_01(x: np.ndarray, mode: str, p_lo: float, p_hi: float) -> tuple[np.ndarray, dict]:
    """
    Returns normalized float32 image in [0..1] and a dict to invert scaling.
    """
    x = np.asarray(x)

    if mode == "dtype":
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            scale = float(info.max)
            y = x.astype(np.float32) / scale
            return np.clip(y, 0.0, 1.0), {"mode": "dtype", "scale": scale}
        # float input -> fallback to minmax
        mode = "minmax"

    if mode == "minmax":
        x_f = x.astype(np.float32)
        lo = float(np.min(x_f))
        hi = float(np.max(x_f))
        if hi <= lo:
            y = np.zeros_like(x_f, dtype=np.float32)
            return y, {"mode": "minmax", "lo": lo, "hi": hi}
        y = (x_f - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0), {"mode": "minmax", "lo": lo, "hi": hi}

    if mode == "percentile":
        x_f = x.astype(np.float32)
        lo = float(np.percentile(x_f, p_lo))
        hi = float(np.percentile(x_f, p_hi))
        if hi <= lo:
            y = np.zeros_like(x_f, dtype=np.float32)
            return y, {"mode": "percentile", "lo": lo, "hi": hi}
        y = (x_f - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0), {"mode": "percentile", "lo": lo, "hi": hi}

    raise ValueError(f"Unknown normalize_mode={mode}")


def _denormalize_from_01(y01: np.ndarray, inv: dict) -> np.ndarray:
    """
    Map [0..1] back to original range.
    """
    y01 = np.asarray(y01, dtype=np.float32)

    if inv["mode"] == "dtype":
        scale = float(inv["scale"])
        return y01 * scale

    if inv["mode"] in ("minmax", "percentile"):
        lo = float(inv["lo"])
        hi = float(inv["hi"])
        if hi <= lo:
            return np.full_like(y01, lo, dtype=np.float32)
        return y01 * (hi - lo) + lo

    raise ValueError(f"Unknown inv mode={inv['mode']}")


def _bm3d_denoise_2d(img2d: np.ndarray, p: BM3DParams) -> np.ndarray:
    """
    Denoise one 2D plane with BM3D, using normalization and optional post-processing.
    """
    bm3d = _require_bm3d()

    # normalize
    img01, inv = _normalize_to_01(img2d, p.normalize_mode, p.percentile_lo, p.percentile_hi)

    # bm3d stage
    stage_arg = bm3d.BM3DStages.ALL_STAGES if p.stage == "all" else bm3d.BM3DStages.HARD_THRESHOLDING

    # Call signature differs slightly across bm3d versions.
    kwargs = dict(sigma_psd=float(p.sigma_psd), stage_arg=stage_arg)
    if p.profile is not None:
        kwargs["profile"] = p.profile

    den01 = bm3d.bm3d(img01, **kwargs).astype(np.float32)

    # optional post Gaussian smoothing (tune: post_gaussian_sigma)
    if p.post_gaussian_sigma and p.post_gaussian_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            den01 = gaussian_filter(den01, sigma=float(p.post_gaussian_sigma)).astype(np.float32)
        except Exception:
            # keep running even if scipy isn't available
            pass

    # optional blending with original (tune: blend_original)
    alpha = float(np.clip(p.blend_original, 0.0, 1.0))
    if alpha < 1.0:
        den01 = (1.0 - alpha) * img01 + alpha * den01

    den01 = np.clip(den01, 0.0, 1.0)

    # rescale back if requested
    if p.preserve_range:
        out = _denormalize_from_01(den01, inv).astype(np.float32)
    else:
        out = den01.astype(np.float32)

    return out


def apply_bm3d_to_image(
    x: np.ndarray,
    axes: str,
    params: BM3DParams,
    *,
    channel_mode: Literal["auto", "blue", "green"] = "auto",
) -> np.ndarray:
    """
    Apply BM3D to a loaded OME-Zarr array (2D dataset: cyx or tcyx).
    Works per (y,x) plane.

    """
    x = np.asarray(x)
    x, axes = _ensure_cyx(x, axes)

    if "c" in axes:
        c_i = axes.index("c")
        n_c = x.shape[c_i]
    else:
        c_i = None
        n_c = 1

    def pick_channels() -> list[int]:
        """Helper function used by this module."""
        if channel_mode == "blue":
            return [0]
        if channel_mode == "green":
            return [1] if n_c > 1 else [0]
        # auto
        return list(range(min(n_c, 2))) if n_c > 1 else [0]

    chs = pick_channels()
    y = x.astype(np.float32, copy=True)

    # Apply per channel and per time frame if present
    if "t" in axes:
        t_i = axes.index("t")
    else:
        t_i = None

    for c in chs:
        sl = [slice(None)] * y.ndim
        if c_i is not None:
            sl[c_i] = c
        plane = y[tuple(sl)]

        if plane.ndim == 2:
            y[tuple(sl)] = _bm3d_denoise_2d(plane, params)
        elif plane.ndim == 3 and t_i is not None:
            # (t,y,x)
            for t in range(plane.shape[0]):
                plane[t] = _bm3d_denoise_2d(plane[t], params)
            y[tuple(sl)] = plane
        else:
            raise ValueError(f"Unexpected plane ndim={plane.ndim} for axes={axes}")

    return y


def run_bm3d_on_dataset(
    dataset: str,
    params: BM3DParams,
    *,
    apply: bool,
    channel_mode: Literal["auto", "blue", "green"] = "auto",
    image_index: int = 0,
    out_subdir_name: str | None = None,
) -> Path:
    """
    Run BM3D on ONE dataset sample (by index) and save OME-Zarr to:
      results/Filters/BM3D/<dataset>/<stem>/image.ome.zarr

    If apply=False, it saves the *original* (no change) to verify the pipeline.
    """
    zarrs = list_omezarr_images(dataset)
    if not zarrs:
        raise FileNotFoundError(f"No OME-Zarr images found for dataset={dataset}")

    image_index = int(np.clip(image_index, 0, len(zarrs) - 1))
    in_path = zarrs[image_index]
    stem = in_path.parent.name

    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)

    if apply:
        y = apply_bm3d_to_image(x, axes, params, channel_mode=channel_mode)
    else:
        y = x.astype(np.float32, copy=True)

    out_root = results_filters_dir() / "BM3D" / dataset
    if out_subdir_name:
        out_root = out_root / out_subdir_name
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # minimal meta 
    n_c = y.shape[axes.index("c")] if "c" in axes else 1
    meta = SimpleNamespace(
        pixel_size_um_x=1.0,
        pixel_size_um_y=1.0,
        pixel_size_um_z=1.0,
        channel_names=[f"ch{i}" for i in range(n_c)],
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