from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PFT.core_prog_parts.io import CziMeta


def normalize01(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8, thr: float = 0.0) -> np.ndarray:
    """Normalize to [0,1]."""
    img = img.astype(np.float32, copy=False)
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        hi = lo + 1e-6
    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if thr and thr > 0:
        out[out < thr] = 0.0
    return out


def max_project_to_2d(arr: np.ndarray) -> np.ndarray:
    """Max-project all dims except last two."""
    if arr.ndim == 2:
        return arr
    reduce_axes = tuple(range(arr.ndim - 2))
    return arr.max(axis=reduce_axes) if reduce_axes else arr


def find_channel_axis(arr: np.ndarray) -> int | None:
    """channel axis detection (2..4 channels)."""
    if arr.ndim < 3:
        return None
    for ax in range(arr.ndim - 2):
        if 2 <= arr.shape[ax] <= 4:
            return ax
    return None


def get_channel_2d(arr: np.ndarray, ch_index: int, channel_axis: int | None = None) -> np.ndarray:
    """Take channel and max-project to 2D."""
    if channel_axis is None:
        channel_axis = find_channel_axis(arr)
    if channel_axis is not None and arr.shape[channel_axis] > 1:
        arr = np.take(arr, indices=ch_index, axis=channel_axis)
    return max_project_to_2d(arr)


def _add_scalebar(ax, meta: CziMeta, length_um: float = 5.0) -> None:
    """Add a scalebar using pixel_size_um_x. If missing, do nothing."""
    sx = meta.pixel_size_um_x
    if sx is None or sx <= 0:
        return

    length_px = length_um / sx
    fontprops = fm.FontProperties(size=10)

    bar = AnchoredSizeBar(
        ax.transData,
        size=length_px,
        label=f"{length_um:g} µm",
        loc="lower right",
        pad=0.4,
        color="white",
        frameon=False,
        size_vertical=max(1.0, length_px * 0.02),
        fontproperties=fontprops,
    )
    ax.add_artist(bar)


def _make_figure_black() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.axis("off")
    return fig, ax


def save_rgb_preview_png(
    rgb01: np.ndarray,
    out_png: Path,
    title: str,
    meta: CziMeta | None = None,
    scalebar_um: float = 4.0,
    dpi: int = 200,
) -> None:
    """
    Save an RGB preview PNG with scalebar.
    rgb01 must be float in [0,1], shape (Y,X,3).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _make_figure_black()
    ax.imshow(rgb01, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")

    if meta is not None:
        _add_scalebar(ax, meta, length_um=scalebar_um)

    # Tight save
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)


def preview_rgb(
    rgb01: np.ndarray,
    title: str,
    meta: CziMeta | None = None,
    scalebar_um: float = 4.0,
) -> None:
    """Show RGB with optional scalebar."""
    fig, ax = _make_figure_black()
    ax.imshow(rgb01, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    if meta is not None:
        _add_scalebar(ax, meta, length_um=scalebar_um)


def rgb_time_hada_blue(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.8, thr: float = 0.0) -> np.ndarray:
    """Single-channel -> BLUE RGB composite."""
    img2d = max_project_to_2d(arr)
    b = normalize01(img2d, p_low=p_low, p_high=p_high, thr=thr)
    return np.dstack([np.zeros_like(b), np.zeros_like(b), b])


def rgb_wga_dapi(arr: np.ndarray, wga_ch: int = 0, dapi_ch: int = 1,
                p_low: float = 1.0, p_high: float = 99.8, thr: float = 0.0) -> np.ndarray:
    """WGA -> GREEN, DAPI -> BLUE."""
    ch_ax = find_channel_axis(arr)
    wga = normalize01(get_channel_2d(arr, wga_ch, ch_ax), p_low=p_low, p_high=p_high, thr=thr)
    dapi = normalize01(get_channel_2d(arr, dapi_ch, ch_ax), p_low=p_low, p_high=p_high, thr=thr)
    return np.dstack([np.zeros_like(wga), wga, dapi])

