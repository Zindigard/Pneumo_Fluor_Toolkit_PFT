"""
Shared visualization helpers used across 2D and 3D processing scripts.

The module creates normalized microscopy previews, RGB channel composites,
FFT displays, scale bars, and comparison figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
if TYPE_CHECKING:
    from PFT.core_prog_parts.io import CziMeta
from PFT.core_prog_parts.image_utils import normalize01_percentile as normalize01

EPS = 1e-12
INTENSITY_RGB_CMAP = LinearSegmentedColormap.from_list(
    "intensity_rgb",
    [
        (0.0, (0.0, 0.0, 0.0)),
        (0.35, (0.0, 0.0, 0.35)),
        (0.65, (0.0, 0.65, 0.0)),
        (1.0, (1.0, 0.0, 0.0)),
    ],
)



def max_project_to_2d(arr: np.ndarray) -> np.ndarray:
    """Helper function used by this module.

    Args:
        arr (np.ndarray): Array containing arr.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = max_project_to_2d(arr=image_array)
    """
    if arr.ndim == 2:
        return arr
    reduce_axes = tuple(range(arr.ndim - 2))
    return arr.max(axis=reduce_axes) if reduce_axes else arr


def find_channel_axis(arr: np.ndarray) -> int | None:
    """Find and return the requested resource.

    Args:
        arr (np.ndarray): Array containing arr.

    Returns:
        int | None: Computed numerical result.

    Example:
        >>> result = find_channel_axis(arr=image_array)
    """
    if arr.ndim < 3:
        return None
    for ax in range(arr.ndim - 2):
        if 2 <= arr.shape[ax] <= 4:
            return ax
    return None


def get_channel_2d(arr: np.ndarray, ch_index: int, channel_axis: int | None = None) -> np.ndarray:
    """Helper function used by this module.

    Args:
        arr (np.ndarray): Array containing arr.
        ch_index (int): Zero-based index selecting ch.
        channel_axis (int | None): Numerical value controlling channel axis. ``None`` selects the function's default behavior.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = get_channel_2d(arr=image_array, ch_index=1)
    """
    if channel_axis is None:
        channel_axis = find_channel_axis(arr)
    if channel_axis is not None and arr.shape[channel_axis] > 1:
        arr = np.take(arr, indices=ch_index, axis=channel_axis)
    return max_project_to_2d(arr)



def make_black_figure(figsize: tuple[float, float] | None = None):
    """Create and return the requested display or object.

    Args:
        figsize (tuple[float, float] | None): Figure width and height, in inches. ``None`` selects the function's default behavior.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = make_black_figure()
    """
    fig = plt.figure(figsize=figsize, facecolor="black")
    return fig


def make_black_axis(figsize: tuple[float, float] | None = None):
    """Create and return a black Matplotlib axis.

    Args:
        figsize (tuple[float, float] | None): Figure width and height, in inches. ``None`` selects the function's default behavior.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = make_black_axis()
    """
    from PFT.core_prog_parts.plot_utils import apply_axis_style

    fig = make_black_figure(figsize=figsize)
    ax = fig.add_subplot(111)
    apply_axis_style(ax, facecolor="black")
    return fig, ax


def save_rgb_preview_png(
    rgb01: np.ndarray,
    out_png: Path,
    title: str,
    meta: CziMeta | None = None,
    scalebar_um: float = 4.0,
    dpi: int = 200,
) -> None:
    """Save generated outputs to disk.

    Args:
        rgb01 (np.ndarray): Array containing rgb01.
        out_png (Path): Filesystem path used for out PNG image.
        title (str): Title displayed on the generated figure or report section.
        meta (CziMeta | None): Value specifying meta for the operation. ``None`` selects the function's default behavior.
        scalebar_um (float): Numerical value controlling scalebar um. Defaults to ``4.0``.
        dpi (int): Resolution of a generated figure in dots per inch. Defaults to ``200``.

    Example:
        >>> save_rgb_preview_png(
        ...     rgb01=image_array,
        ...     out_png=Path("path/to/resource"),
        ...     title="title",
        ... )
    """
    from PFT.core_prog_parts.plot_utils import add_scalebar, finalize_figure

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = make_black_axis()
    ax.imshow(rgb01, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    if meta is not None:
        add_scalebar(ax, meta, length_um=scalebar_um)
    finalize_figure(fig, out_png, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0.05)


def preview_rgb(rgb01: np.ndarray, title: str, meta: CziMeta | None = None, scalebar_um: float = 4.0) -> None:
    """Display an RGB preview, optionally with a scale bar.

    Args:
        rgb01 (np.ndarray): Array containing rgb01.
        title (str): Title displayed on the generated figure or report section.
        meta (CziMeta | None): Value specifying meta for the operation. ``None`` selects the function's default behavior.
        scalebar_um (float): Numerical value controlling scalebar um. Defaults to ``4.0``.

    Example:
        >>> preview_rgb(rgb01=image_array, title="title")
    """
    from PFT.core_prog_parts.plot_utils import add_scalebar

    fig, ax = make_black_axis()
    ax.imshow(rgb01, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    if meta is not None:
        add_scalebar(ax, meta, length_um=scalebar_um)


def _normalize_preview_channel(
    image: np.ndarray,
    *,
    p_low: float,
    p_high: float,
    thr: float,
) -> np.ndarray:
    """Normalize one preview channel and optionally suppress weak values.

    Args:
        image (np.ndarray): Input image or projected channel.
        p_low (float): Lower and upper percentiles used for clipping and normalization.
        p_high (float): Lower and upper percentiles used for clipping and normalization.
        thr (float): Optional threshold in the normalized ``[0, 1]`` range. Values below the threshold are set to zero. A value of ``0`` disables thresholding.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _normalize_preview_channel(
        ...     image=image_array,
        ...     p_low=0.5,
        ...     p_high=0.5,
        ...     thr=0.5,
        ... )
    """
    normalized = normalize01(image, p_lo=p_low, p_hi=p_high)
    if thr > 0.0:
        normalized = np.where(normalized >= thr, normalized, 0.0).astype(
            np.float32, copy=False
        )
    return normalized


def rgb_time_hada_blue(
    arr: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.8,
    thr: float = 0.0,
) -> np.ndarray:
    """Create a blue RGB preview from a time-series or single-channel image.

    Args:
        arr (np.ndarray): Array containing arr.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.
        thr (float): Numerical value controlling thr. Defaults to ``0.0``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_time_hada_blue(arr=image_array)
    """
    img2d = max_project_to_2d(arr)
    b = _normalize_preview_channel(
        img2d, p_low=p_low, p_high=p_high, thr=thr
    )
    return np.dstack([np.zeros_like(b), np.zeros_like(b), b])


def rgb_wga_dapi(
    arr: np.ndarray,
    wga_ch: int = 0,
    dapi_ch: int = 1,
    p_low: float = 1.0,
    p_high: float = 99.8,
    thr: float = 0.0,
) -> np.ndarray:
    """Create a green WGA and blue DAPI RGB preview from a multichannel image.

    Args:
        arr (np.ndarray): Array containing arr.
        wga_ch (int): Numerical value controlling wga ch. Defaults to ``0``.
        dapi_ch (int): Numerical value controlling dapi ch. Defaults to ``1``.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.
        thr (float): Numerical value controlling thr. Defaults to ``0.0``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_wga_dapi(arr=image_array)
    """
    ch_ax = find_channel_axis(arr)
    wga = _normalize_preview_channel(
        get_channel_2d(arr, wga_ch, ch_ax),
        p_low=p_low,
        p_high=p_high,
        thr=thr,
    )
    dapi = _normalize_preview_channel(
        get_channel_2d(arr, dapi_ch, ch_ax),
        p_low=p_low,
        p_high=p_high,
        thr=thr,
    )
    return np.dstack([np.zeros_like(wga), wga, dapi])


def make_crimson_cmap():
    """Create and return the requested display or object.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = make_crimson_cmap()
    """
    return LinearSegmentedColormap.from_list(
        "black_to_crimson",
        [
            (0.0, (0.0, 0.0, 0.0)),
            (0.10, (0.08, 0.0, 0.0)),
            (0.35, (0.30, 0.0, 0.04)),
            (0.65, (0.62, 0.02, 0.10)),
            (1.0, (0.86, 0.08, 0.24)),
        ],
    )


def fft_log_magnitude(img2d: np.ndarray) -> np.ndarray:
    """Helper function used by this module.

    Args:
        img2d (np.ndarray): Array containing img2d.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = fft_log_magnitude(img2d=image_array)
    """
    x = np.asarray(img2d, dtype=np.float32)
    x = x - float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)
    vis = np.log1p(mag)
    lo = float(np.percentile(vis, 1.0))
    hi = float(np.percentile(vis, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((vis - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def imshow_percentile(ax, img: np.ndarray, cmap: str = "gray", p_low: float = 1.0, p_high: float = 99.7) -> None:
    """Helper function used by this module.

    Args:
        ax (Any): Matplotlib axes object on which graphical elements are drawn.
        img (np.ndarray): Array containing img.
        cmap (str): Matplotlib colormap used to display scalar image intensities. Defaults to ``"gray"``.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.7``.

    Example:
        >>> imshow_percentile(ax=..., img=image_array)
    """
    vmin = float(np.percentile(img, p_low))
    vmax = float(np.percentile(img, p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)


def shared_display_limits(
    reference: np.ndarray,
    *,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> tuple[float, float]:
    """Return robust display limits derived only from the original image.

    This function is intended for visualization. It does not modify the
    quantitative image and must not be used to rescale OME-Zarr output.

    Args:
        reference (np.ndarray): Array containing reference.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        tuple[float, float]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = shared_display_limits(reference=image_array)
    """
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("Display percentiles must satisfy 0 <= p_low < p_high <= 100")
    values = np.asarray(reference, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (p_low, p_high))
    low = float(low)
    high = float(high)
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return low, high


def normalize_display_with_limits(
    image: np.ndarray,
    limits: tuple[float, float],
) -> np.ndarray:
    """Map an image to ``[0, 1]`` for display using fixed shared limits.

    Args:
        image (np.ndarray): Input image array to process.
        limits (tuple[float, float]): Numerical value controlling limits.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = normalize_display_with_limits(image=image_array, limits=0.5)
    """
    low, high = (float(limits[0]), float(limits[1]))
    if high <= low:
        raise ValueError("Display upper limit must be larger than the lower limit")
    values = np.asarray(image, dtype=np.float32)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_original_filtered_pair(
    original: np.ndarray,
    filtered: np.ndarray,
    *,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Normalize original and filtered planes with one original-derived scale.

    The two inputs must have the same shape. Shared normalization ensures that
    apparent contrast changes reflect filtering rather than independent display
    stretching. The returned arrays are display products only.

    Args:
        original (np.ndarray): Array containing original.
        filtered (np.ndarray): Array containing filtered.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[float, float]]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = normalize_original_filtered_pair(original=image_array, filtered=image_array)
    """
    original_array = np.asarray(original)
    filtered_array = np.asarray(filtered)
    if original_array.shape != filtered_array.shape:
        raise ValueError(
            f"Original and filtered shapes differ: {original_array.shape} vs "
            f"{filtered_array.shape}"
        )
    limits = shared_display_limits(original_array, p_low=p_low, p_high=p_high)
    return (
        normalize_display_with_limits(original_array, limits),
        normalize_display_with_limits(filtered_array, limits),
        limits,
    )


def save_normalized_before_after_plot(
    original: np.ndarray,
    filtered: np.ndarray,
    out_png: Path,
    *,
    title: str = "Original versus filtered",
    original_label: str = "Original",
    filtered_label: str = "Filtered",
    p_low: float = 1.0,
    p_high: float = 99.8,
    cmap: str = "gray",
    dpi: int = 200,
    footer: str | None = None,
) -> tuple[float, float]:
    """Save a two-panel, shared-normalization comparison plot.

    Both panels use percentile limits calculated from ``original``. The plot is
    a display-only product. Source and filtered quantitative arrays are not
    modified or saved by this function. The applied raw-intensity limits are
    returned for provenance.

    Args:
        original (np.ndarray): Array containing original.
        filtered (np.ndarray): Array containing filtered.
        out_png (Path): Filesystem path used for out PNG image.
        title (str): Title displayed on the generated figure or report section. Defaults to ``"Original versus filtered"``.
        original_label (str): Text value specifying original label. Defaults to ``"Original"``.
        filtered_label (str): Text value specifying filtered label. Defaults to ``"Filtered"``.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.
        cmap (str): Matplotlib colormap used to display scalar image intensities. Defaults to ``"gray"``.
        dpi (int): Resolution of a generated figure in dots per inch. Defaults to ``200``.
        footer (str | None): Text value specifying footer. ``None`` selects the function's default behavior.

    Returns:
        tuple[float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = save_normalized_before_after_plot(
        ...     original=image_array,
        ...     filtered=image_array,
        ...     out_png=Path("path/to/resource"),
        ... )
    """
    original_display, filtered_display, limits = normalize_original_filtered_pair(
        original,
        filtered,
        p_low=p_low,
        p_high=p_high,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for axis, image, panel_title in (
        (axes[0], original_display, original_label),
        (axes[1], filtered_display, filtered_label),
    ):
        axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title)
    if footer:
        figure.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return limits


def plot_rgb_comparison(orig_rgb: np.ndarray, filt_rgb: np.ndarray, title: str, out_png: Path, *, original_label: str = "Original", filtered_label: str = "Filtered", facecolor: str = "black") -> None:
    """Helper function used by this module.

    Args:
        orig_rgb (np.ndarray): Array containing orig RGB representation.
        filt_rgb (np.ndarray): Array containing filt RGB representation.
        title (str): Title displayed on the generated figure or report section.
        out_png (Path): Filesystem path used for out PNG image.
        original_label (str): Text value specifying original label. Defaults to ``"Original"``.
        filtered_label (str): Text value specifying filtered label. Defaults to ``"Filtered"``.
        facecolor (str): Text value specifying facecolor. Defaults to ``"black"``.

    Example:
        >>> plot_rgb_comparison(
        ...     orig_rgb=image_array,
        ...     filt_rgb=image_array,
        ...     title="title",
        ...     out_png=Path("path/to/resource"),
        ... )
    """
    diff = np.abs(filt_rgb.astype(np.float32) - orig_rgb.astype(np.float32))
    diff_map = np.mean(diff, axis=-1)
    scale = float(np.percentile(diff_map, 99.7)) + EPS
    diff_norm = np.clip(diff_map / scale, 0.0, 1.0)

    fig = plt.figure(figsize=(12, 4), facecolor=facecolor)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    for ax in (ax1, ax2, ax3):
        ax.axis("off")
    ax1.imshow(orig_rgb)
    ax1.set_title(original_label)
    ax2.imshow(filt_rgb)
    ax2.set_title(filtered_label)
    ax3.imshow(diff_norm, cmap=make_crimson_cmap(), vmin=0, vmax=1)
    ax3.set_title("|Δ| (black=min, crimson=max)")
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)


def save_fft_with_dc_overlay(fft_img: np.ndarray, r_dc: int, out_png: Path, title: str, cmap: str, linewidth: int = 3, overlay_alpha: float = 0.25) -> None:
    """Save generated outputs to disk.

    Args:
        fft_img (np.ndarray): Array containing Fourier-transform result img.
        r_dc (int): Numerical value controlling r dc.
        out_png (Path): Filesystem path used for out PNG image.
        title (str): Title displayed on the generated figure or report section.
        cmap (str): Matplotlib colormap used to display scalar image intensities.
        linewidth (int): Numerical value controlling linewidth. Defaults to ``3``.
        overlay_alpha (float): Numerical value controlling overlay alpha. Defaults to ``0.25``.

    Example:
        >>> save_fft_with_dc_overlay(
        ...     fft_img=image_array,
        ...     r_dc=1,
        ...     out_png=Path("path/to/resource"),
        ...     title="title",
        ...     cmap="cmap",
        ... )
    """
    h, w = fft_img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    fig, ax = plt.subplots()
    imshow_percentile(ax, fft_img, cmap=cmap)
    ax.axis("off")
    ax.set_title(title)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.55
    overlay[..., 2] = 0.0
    ax.imshow(overlay, alpha=(rr <= float(r_dc)).astype(np.float32) * overlay_alpha)
    ax.add_patch(Circle((cx, cy), r_dc, fill=False, edgecolor="orange", linewidth=linewidth))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_single_channel_outputs(channel: np.ndarray, out_dir: Path, stem: str, channel_name: str, *, cmap=INTENSITY_RGB_CMAP) -> None:
    """Save generated outputs to disk.

    Args:
        channel (np.ndarray): Channel index or channel identifier selected for processing.
        out_dir (Path): Directory used for out.
        stem (str): Text value specifying stem.
        channel_name (str): Text value specifying channel name.
        cmap (Any): Matplotlib colormap used to display scalar image intensities. Defaults to ``INTENSITY_RGB_CMAP``.

    Example:
        >>> save_single_channel_outputs(
        ...     channel=image_array,
        ...     out_dir=Path("path/to/resource"),
        ...     stem="stem",
        ...     channel_name="channel_name",
        ... )
    """
    intensity = normalize01(channel)
    fft_img = fft_log_magnitude(channel)
    plt.imsave(out_dir / f"{stem}_{channel_name}_intensity_map.png", intensity, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.imsave(out_dir / f"{stem}_{channel_name}_fft.png", fft_img, cmap="gray", vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")
    axes[0].imshow(intensity, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title(f"{channel_name} intensity", color="white")
    axes[1].imshow(fft_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{channel_name} FFT", color="white")
    fig.suptitle(stem, color="white")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_{channel_name}_intensity_fft_panel.png", dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def save_rgb_overview(rgb: np.ndarray, out_dir: Path, stem: str, *, panel_title: str | None = None) -> None:
    """Save generated outputs to disk.

    Args:
        rgb (np.ndarray): Array containing RGB representation.
        out_dir (Path): Directory used for out.
        stem (str): Text value specifying stem.
        panel_title (str | None): Text value specifying panel title. ``None`` selects the function's default behavior.

    Example:
        >>> save_rgb_overview(
        ...     rgb=image_array,
        ...     out_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
    plt.imsave(out_dir / f"{stem}_normalized_rgb.png", rgb)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgb)
    ax.set_title(panel_title or f"{stem} normalized RGB", color="white")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_normalized_rgb_panel.png", dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure_tiff(raw_rgb_uint8: np.ndarray, norm_rgb_float: np.ndarray, out_path: Path, title: str) -> None:
    """Save generated outputs to disk.

    Args:
        raw_rgb_uint8 (np.ndarray): Array containing raw RGB representation uint8.
        norm_rgb_float (np.ndarray): Array containing norm RGB representation float.
        out_path (Path): Filesystem path associated with out.
        title (str): Title displayed on the generated figure or report section.

    Example:
        >>> save_comparison_figure_tiff(
        ...     raw_rgb_uint8=image_array,
        ...     norm_rgb_float=image_array,
        ...     out_path=Path("path/to/resource"),
        ...     title="title",
        ... )
    """
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(raw_rgb_uint8)
    ax1.set_title("RAW display (linear -> uint8)")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(norm_rgb_float)
    ax2.set_title("NORMALIZED display (p1-p99)")
    ax2.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="tiff", dpi=200)
    plt.close(fig)


def save_image_and_fft(img: np.ndarray, disp: np.ndarray, out_dir: Path, stem: str, *, save_image_png: bool = True) -> None:
    """Save generated outputs to disk.

    Args:
        img (np.ndarray): Array containing img.
        disp (np.ndarray): Array containing disp.
        out_dir (Path): Directory used for out.
        stem (str): Text value specifying stem.
        save_image_png (bool): Boolean flag controlling save image PNG image. Defaults to ``True``.

    Example:
        >>> save_image_and_fft(
        ...     img=image_array,
        ...     disp=image_array,
        ...     out_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_image_png:
        img_path = out_dir / f"{stem}_img.png"
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
    fft_path = out_dir / f"{stem}_fft2d.png"
    plt.figure()
    plt.imshow(disp)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(fft_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_unet_overlay_panel(rgb: np.ndarray, mask_yx: np.ndarray, overlay: np.ndarray, out_png: Path, *, input_title: str = "Input image", mask_title: str = "Predicted mask", overlay_title: str = "Overlay with outlines") -> None:
    """Save generated outputs to disk.

    Args:
        rgb (np.ndarray): Array containing RGB representation.
        mask_yx (np.ndarray): Array containing mask yx.
        overlay (np.ndarray): Array containing overlay.
        out_png (Path): Filesystem path used for out PNG image.
        input_title (str): Text value specifying input title. Defaults to ``"Input image"``.
        mask_title (str): Text value specifying mask title. Defaults to ``"Predicted mask"``.
        overlay_title (str): Text value specifying overlay title. Defaults to ``"Overlay with outlines"``.

    Example:
        >>> save_unet_overlay_panel(
        ...     rgb=image_array,
        ...     mask_yx=image_array,
        ...     overlay=image_array,
        ...     out_png=Path("path/to/resource"),
        ... )
    """
    fig = plt.figure(figsize=(14, 4.5), dpi=180)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title(input_title)
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(mask_yx, cmap="gray")
    ax2.set_title(mask_title)
    ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title(overlay_title)
    ax3.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def show_two_channel_overlay(c0n: np.ndarray, c1n: np.ndarray, rgb: np.ndarray, title: str) -> None:
    """Display a quick visual preview for inspection.

    Args:
        c0n (np.ndarray): Array containing c0n.
        c1n (np.ndarray): Array containing c1n.
        rgb (np.ndarray): Array containing RGB representation.
        title (str): Title displayed on the generated figure or report section.

    Example:
        >>> show_two_channel_overlay(
        ...     c0n=image_array,
        ...     c1n=image_array,
        ...     rgb=image_array,
        ...     title="title",
        ... )
    """
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(c0n, cmap="gray")
    ax1.set_title("Channel 0 (grayscale)")
    ax1.axis("off")
    ax2.imshow(c1n, cmap="gray")
    ax2.set_title("Channel 1 (grayscale)")
    ax2.axis("off")
    ax3.imshow(rgb)
    ax3.set_title("Overlay: ch0=Blue, ch1=Green")
    ax3.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def main_visualization_demo(arr: np.ndarray, dataset: str = "2d_time") -> np.ndarray:
    """Return a normalized RGB preview for the main supported 2D datasets.

    Args:
        arr (np.ndarray): Input array. For ``2d_time`` this is treated as a single-channel image or stack. For ``2d_wga_dapi`` this is treated as a 2-channel array/stack.
        dataset (str): One of ``"2d_time"`` or ``"2d_wga_dapi"``.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = main_visualization_demo(arr=image_array)
    """
    if dataset == "2d_time":
        return rgb_time_hada_blue(arr)
    if dataset == "2d_wga_dapi":
        return rgb_wga_dapi(arr)
    raise ValueError(f"Unsupported dataset: {dataset}")
