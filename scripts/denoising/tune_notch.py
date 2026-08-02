r"""Interactive tuning of the optional directional notch filter.

The script visualizes mean FFTs and lets the user select angular stop bands.
It is retained for comparison experiments and is not the default 2D filter.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/tune_notch.py --help

Representative execution:

    python scripts/denoising/tune_notch.py \
        --dataset 2d_time \
        --fft_size 512 \
        --image_index 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from PFT.core_prog_parts import visualization as viz
from PFT.core_prog_parts.image_utils import extract_display_plane as _extract_display_plane
import numpy as np
from matplotlib.patches import Circle

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.denoising.notch_filter import (
    NotchParams,
    list_omezarr_images,
    results_filters_dir,
    run_notch_on_dataset,
)
from PFT.core_prog_parts.denoising.notch_filter import _ensure_cyx, _to_numpy

EPS = 1e-12



CURATED_TEST_STEMS: dict[str, list[str]] = {
    "2d_time": [
        "WT_HADA_NHS_5min_ROI1_SIM²",
        "WT_HADA_NHS_5min_ROI4_SIM",
        "WT_HADA_NHS_20min_ROI1_SIM",
        "WT_HADA_NHS_20min_ROI2_SIM",
        "WT_HADA_NHS_20min_ROI3_SIM",
        "WT_HADA_NHS_120min_ROI1_SIM",
        "WT_HADA_NHS_40min_ROI1_SIM²",
        "WT_HADA_NHS_120min_ROI3_SIM²",
        "WT_HADA_THY_20min_ROI2_SIM²",
        "WT_HADA_THY_40min_ROI2_SIM²",
        "WT_HADA_THY_40min_ROI4_SIM²",
        "WT_HADA_THY_120min_ROI1_SIM²",
    ]
}


def list_curated_test_images(dataset: str) -> list[Path]:
    """Return workflow images.

    - 2d_time: use the fixed curated subset
    - 2d_wga_dapi: use ALL available images

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = list_curated_test_images(dataset="2d_time")
    """
    all_imgs = list_omezarr_images(dataset)

    if dataset == "2d_wga_dapi":
        return all_imgs

    want = CURATED_TEST_STEMS.get(dataset, [])
    by_stem = {p.parent.name: p for p in all_imgs}
    return [by_stem[s] for s in want if s in by_stem]


def missing_curated_test_stems(dataset: str) -> list[str]:
    """Return missing curated stems.

    For 2d_wga_dapi we use all images, so nothing is missing.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[str]: Collection containing the generated or selected values.

    Example:
        >>> result = missing_curated_test_stems(dataset="2d_time")
    """
    if dataset == "2d_wga_dapi":
        return []

    all_stems = {p.parent.name for p in list_omezarr_images(dataset)}
    return [s for s in CURATED_TEST_STEMS.get(dataset, []) if s not in all_stems]


# -------------------------
# Metrics
# -------------------------
def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return pearson corr for the supplied inputs.

    Args:
        a (np.ndarray): Array containing a.
        b (np.ndarray): Array containing b.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = _pearson_corr(a=image_array, b=image_array)
    """
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS
    return float(np.sum(a * b) / denom)


def neighbor_corr(x: np.ndarray) -> float:
    """Return neighbor corr for the supplied inputs.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = neighbor_corr(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    vals = []
    if x.shape[1] >= 2:
        vals.append(_pearson_corr(x[:, :-1], x[:, 1:]))
    if x.shape[0] >= 2:
        vals.append(_pearson_corr(x[:-1, :], x[1:, :]))
    return float(np.mean(vals)) if vals else 0.0


def fft_peak_score(x: np.ndarray, dc_halfwidth: int = 8) -> float:
    """Return Fourier-transform result peak score for the supplied inputs.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        dc_halfwidth (int): Numerical value controlling dc halfwidth. Defaults to ``8``.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = fft_peak_score(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F).astype(np.float64)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - dc_halfwidth), min(h, cy + dc_halfwidth + 1)
    x0, x1 = max(0, cx - dc_halfwidth), min(w, cx + dc_halfwidth + 1)
    mag[y0:y1, x0:x1] = 0.0

    mean = float(mag.mean()) + EPS
    p999 = float(np.percentile(mag, 99.9))
    return p999 / mean


def gradient_mag_mean(x: np.ndarray) -> float:
    """Return gradient mag mean for the supplied inputs.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = gradient_mag_mean(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    gy, gx = np.gradient(x)
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g))


def compute_metrics_per_channel(blue: np.ndarray, green: np.ndarray | None) -> dict:
    """Compute metrics per channel from the supplied data.

    Args:
        blue (np.ndarray): Array containing blue.
        green (np.ndarray | None): Array containing green.

    Returns:
        dict: Mapping containing the generated or resolved values.

    Example:
        >>> result = compute_metrics_per_channel(blue=image_array, green=image_array)
    """
    out = {
        "blue": {
            "neighbor_corr": neighbor_corr(blue),
            "fft_peak_score": fft_peak_score(blue),
            "grad_mag_mean": gradient_mag_mean(blue),
        }
    }
    if green is not None:
        out["green"] = {
            "neighbor_corr": neighbor_corr(green),
            "fft_peak_score": fft_peak_score(green),
            "grad_mag_mean": gradient_mag_mean(green),
        }
    return out


def write_metrics_block(f, title: str, metrics: dict) -> None:
    """Write metrics block to persistent storage.

    Args:
        f (Any): Value specifying f for the operation.
        title (str): Title displayed on the generated figure or report section.
        metrics (dict): Value specifying metrics for the operation.

    Example:
        >>> write_metrics_block(
        ...     f=...,
        ...     title="title",
        ...     metrics={},
        ... )
    """
    f.write(f"\n=== {title} ===\n")
    for ch_name, m in metrics.items():
        f.write(f"\n[{ch_name}]\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")


# -------------------------
# Display helpers
# -------------------------
def _norm01_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    """Return norm01 percentile for the supplied inputs.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        p_lo (float): Numerical value controlling p lo. Defaults to ``1.0``.
        p_hi (float): Numerical value controlling p hi. Defaults to ``99.5``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _norm01_percentile(x=image_array)
    """
    x = x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _to_rgb_from_blue_green(blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    """Return to RGB representation from blue green for the supplied inputs.

    Args:
        blue (np.ndarray): Array containing blue.
        green (np.ndarray | None): Array containing green.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _to_rgb_from_blue_green(blue=image_array, green=image_array)
    """
    B = _norm01_percentile(blue)
    G = _norm01_percentile(green) if green is not None else np.zeros_like(B)
    R = np.zeros_like(B)
    return np.stack([R, G, B], axis=-1)


def _make_crimson_cmap():
    """Create crimson cmap from the supplied inputs.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = _make_crimson_cmap()
    """
    return viz.make_crimson_cmap()


def _plot_rgb_comparison(orig_rgb: np.ndarray, filt_rgb: np.ndarray, title: str, out_png: Path):
    """Plot RGB representation comparison for visual assessment.

    Args:
        orig_rgb (np.ndarray): Array containing orig RGB representation.
        filt_rgb (np.ndarray): Array containing filt RGB representation.
        title (str): Title displayed on the generated figure or report section.
        out_png (Path): Filesystem path used for out PNG image.

    Example:
        >>> _plot_rgb_comparison(
        ...     orig_rgb=image_array,
        ...     filt_rgb=image_array,
        ...     title="title",
        ...     out_png=Path("path/to/resource"),
        ... )
    """
    viz.plot_rgb_comparison(
        orig_rgb,
        filt_rgb,
        title,
        out_png,
        original_label="Original (RGB, display-norm)",
        filtered_label="Filtered (RGB, display-norm)",
    )



def _fft_logmag(img2d: np.ndarray) -> np.ndarray:
    """Return Fourier-transform result logmag for the supplied inputs.

    Args:
        img2d (np.ndarray): Array containing img2d.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _fft_logmag(img2d=image_array)
    """
    x = img2d.astype(np.float32, copy=False)
    x = x - float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x))
    return np.log1p(np.abs(F)).astype(np.float32)


def _center_crop(img2d: np.ndarray, target: int = 512) -> np.ndarray:
    """Return center crop for the supplied inputs.

    Args:
        img2d (np.ndarray): Array containing img2d.
        target (int): Numerical value controlling target. Defaults to ``512``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _center_crop(img2d=image_array)
    """
    h, w = img2d.shape
    if min(h, w) <= target:
        return img2d
    y0 = (h - target) // 2
    x0 = (w - target) // 2
    return img2d[y0:y0 + target, x0:x0 + target]


def mean_fft_magnitude(
    zarr_paths: list[Path],
    *,
    channel_index: int,
    fft_size: int = 512,
) -> np.ndarray:
    """Return mean Fourier-transform result magnitude for the supplied inputs.

    Args:
        zarr_paths (list[Path]): Filesystem path used for Zarr paths.
        channel_index (int): Zero-based index selecting channel.
        fft_size (int): Size parameter controlling Fourier-transform result. Defaults to ``512``.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = mean_fft_magnitude(zarr_paths=Path("path/to/resource"), channel_index=1)
    """
    if not zarr_paths:
        raise ValueError("No images to compute mean FFT.")

    acc = None
    used = 0
    for p in zarr_paths:
        arr, axes = load_ome_zarr(p, level=0, as_numpy=False)
        x = _to_numpy(arr)
        x, axes = _ensure_cyx(x, axes)

        if "c" in axes:
            c_i = axes.index("c")
            if channel_index >= x.shape[c_i]:
                continue
            img = np.take(x, indices=channel_index, axis=c_i)
        else:
            img = x

        if "t" in axes and img.ndim == 3:
            img = img[0]
        if img.ndim != 2:
            continue

        img = _center_crop(img, fft_size)
        m = _fft_logmag(img)
        acc = m.astype(np.float64) if acc is None else (acc + m.astype(np.float64))
        used += 1

    if acc is None or used == 0:
        raise ValueError("Could not compute mean FFT (no valid images).")
    return (acc / float(used)).astype(np.float32)


def angle_energy_profile(
    mean_logmag: np.ndarray,
    *,
    r_min: int,
    r_max: int | None = None,
    n_bins: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Return angle energy profile for the supplied inputs.

    Args:
        mean_logmag (np.ndarray): Array containing mean logmag.
        r_min (int): Numerical value controlling r min.
        r_max (int | None): Numerical value controlling r max. ``None`` selects the function's default behavior.
        n_bins (int): Number of bins used by the operation. Defaults to ``360``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = angle_energy_profile(mean_logmag=image_array, r_min=1)
    """
    h, w = mean_logmag.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.degrees(np.arctan2(yy - cy, xx - cx))
    ang = (ang + 360.0) % 360.0

    if r_max is None:
        r_max = int(rr.max())
    valid = (rr >= float(r_min)) & (rr <= float(r_max))

    flat = mean_logmag.astype(np.float32) - float(np.median(mean_logmag))
    flat = np.clip(flat, 0.0, None)

    bins = np.linspace(0.0, 360.0, n_bins + 1)
    prof = np.zeros(n_bins, dtype=np.float64)

    a = ang[valid].ravel()
    v = flat[valid].ravel().astype(np.float64)

    idx = np.searchsorted(bins, a, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    np.add.at(prof, idx, v)

    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers.astype(np.float32), prof.astype(np.float64)


def _channel_cmap(name: str) -> str:
    """Return channel cmap for the supplied inputs.

    Args:
        name (str): Name used to identify the current object, resource, or output.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _channel_cmap(name="name")
    """
    return {"blue": "Blues", "green": "Greens"}.get(name, "gray")


def _imshow_fft(ax, fft_img: np.ndarray, cmap: str):
    """Return imshow Fourier-transform result for the supplied inputs.

    Args:
        ax (Any): Matplotlib axes object on which graphical elements are drawn.
        fft_img (np.ndarray): Array containing Fourier-transform result img.
        cmap (str): Matplotlib colormap used to display scalar image intensities.

    Example:
        >>> _imshow_fft(
        ...     ax=...,
        ...     fft_img=image_array,
        ...     cmap="cmap",
        ... )
    """
    viz.imshow_percentile(ax, fft_img, cmap=cmap)


def _pick_radius_centered(mean_fft: np.ndarray, title: str, cmap: str) -> int:
    """Return pick radius centered for the supplied inputs.

    Args:
        mean_fft (np.ndarray): Array containing mean Fourier-transform result.
        title (str): Title displayed on the generated figure or report section.
        cmap (str): Matplotlib colormap used to display scalar image intensities.

    Returns:
        int: Computed numerical result.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _pick_radius_centered(
        ...     mean_fft=image_array,
        ...     title="title",
        ...     cmap="cmap",
        ... )
    """
    h, w = mean_fft.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    fig, ax = plt.subplots()
    _imshow_fft(ax, mean_fft, cmap=cmap)
    ax.set_title(title + "\nMove mouse (red preview), click once to set DC radius")
    ax.axis("off")

    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.55
    overlay[..., 2] = 0.0
    overlay_img = ax.imshow(overlay, alpha=0.0)

    circ = Circle((cx, cy), 1.0, fill=False, edgecolor="red", linewidth=1)
    ax.add_patch(circ)

    selected = {"r": None}

    def on_move(event):
        """Handle the move callback event.

        Args:
            event (Any): Event object supplied by the graphical user interface or callback framework.

        Example:
            >>> on_move(event=event)
        """
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        r = float(np.sqrt((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2))
        circ.set_radius(r)
        overlay_img.set_alpha((rr <= r).astype(np.float32) * 0.25)
        fig.canvas.draw_idle()

    def on_click(event):
        """Handle the click callback event.

        Args:
            event (Any): Event object supplied by the graphical user interface or callback framework.

        Example:
            >>> on_click(event=event)
        """
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        r = float(np.sqrt((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2))
        selected["r"] = max(1, int(round(r)))
        plt.close(fig)

    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)
    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_click)

    if selected["r"] is None:
        raise RuntimeError("DC radius selection cancelled.")
    return int(selected["r"])


def _save_fft_with_dc_overlay(fft_img: np.ndarray, r_dc: int, out_png: Path, title: str, cmap: str, linewidth: int = 3):
    """Save Fourier-transform result with dc overlay to persistent storage.

    Args:
        fft_img (np.ndarray): Array containing Fourier-transform result img.
        r_dc (int): Numerical value controlling r dc.
        out_png (Path): Filesystem path used for out PNG image.
        title (str): Title displayed on the generated figure or report section.
        cmap (str): Matplotlib colormap used to display scalar image intensities.
        linewidth (int): Numerical value controlling linewidth. Defaults to ``3``.

    Example:
        >>> _save_fft_with_dc_overlay(
        ...     fft_img=image_array,
        ...     r_dc=1,
        ...     out_png=Path("path/to/resource"),
        ...     title="title",
        ...     cmap="cmap",
        ... )
    """
    viz.save_fft_with_dc_overlay(fft_img, r_dc, out_png, title, cmap, linewidth=linewidth)



def _prompt_floats(prompt: str, default: list[float]) -> list[float]:
    """Return prompt floats for the supplied inputs.

    Args:
        prompt (str): Text value specifying prompt.
        default (list[float]): Numerical value controlling default.

    Returns:
        list[float]: Collection containing the generated or selected values.

    Example:
        >>> result = _prompt_floats(prompt="prompt", default=0.5)
    """
    s = input(prompt).strip()
    if not s:
        return default
    s = s.replace(",", " ")
    return [float(tok) for tok in s.split()]


def _prompt_float(prompt: str, default: float) -> float:
    """Return prompt float for the supplied inputs.

    Args:
        prompt (str): Text value specifying prompt.
        default (float): Numerical value controlling default.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = _prompt_float(prompt="prompt", default=0.5)
    """
    s = input(prompt).strip()
    return default if not s else float(s)


def _prompt_int(prompt: str, default: int) -> int:
    """Return prompt int for the supplied inputs.

    Args:
        prompt (str): Text value specifying prompt.
        default (int): Numerical value controlling default.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _prompt_int(prompt="prompt", default=1)
    """
    s = input(prompt).strip()
    return default if not s else int(s)


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    """Return prompt bool for the supplied inputs.

    Args:
        prompt (str): Text value specifying prompt.
        default (bool): Boolean flag controlling default. Defaults to ``False``.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _prompt_bool(prompt="prompt")
    """
    s = input(prompt).strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true", "t")


def _choose_dataset_interactive() -> str:
    """Choose dataset interactive according to the configured criteria.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _choose_dataset_interactive()
    """
    print("Choose dataset:")
    print("  1) 2d_time (blue only)")
    print("  2) 2d_wga_dapi (blue + green)")
    ans = input("Choose number [1/2]: ").strip() or "1"
    return "2d_wga_dapi" if ans == "2" else "2d_time"


def _choose_mode_interactive() -> str:
    """Choose mode interactive according to the configured criteria.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _choose_mode_interactive()
    """
    print("Choose mode:")
    print("  1) Process ONE image directly (enter parameters first, skip tuning figures)")
    print("  2) Tune on curated subset first, then process curated subset")
    print("  3) Process curated subset directly (enter parameters first, skip tuning figures)")
    ans = input("Choose number [1/2/3]: ").strip() or "2"
    return ans if ans in {"1", "2", "3"} else "2"


def _path_to_index(all_paths: list[Path], target: Path) -> int:
    """Return path to index for the supplied inputs.

    Args:
        all_paths (list[Path]): Filesystem path used for all paths.
        target (Path): Filesystem path used for target.

    Returns:
        int: Computed numerical result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _path_to_index(all_paths=Path("path/to/resource"), target=Path("path/to/resource"))
    """
    target_resolved = target.resolve()
    for i, p in enumerate(all_paths):
        if p.resolve() == target_resolved:
            return i
    raise ValueError(f"Path not found in dataset list: {target}")


def _extract_display_plane(x: np.ndarray, axes: str, c: int) -> np.ndarray:
    """Extract display plane from the supplied data.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        axes (str): Axis specification describing the dimensional order of the image data.
        c (int): Numerical value controlling c.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _extract_display_plane(
        ...     x=image_array,
        ...     axes="axes",
        ...     c=1,
        ... )
    """
    arr = x
    if "c" in axes:
        arr = np.take(arr, indices=c, axis=axes.index("c"))
    if "t" in axes and arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D display plane, got shape={arr.shape}, axes={axes}")
    return np.asarray(arr, dtype=np.float32)


def _collect_dataset_channel_info(dataset: str, zarrs: list[Path]) -> tuple[list[int], list[str]]:
    """Collect dataset channel info from the available inputs.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        zarrs (list[Path]): Filesystem path used for zarrs.

    Returns:
        tuple[list[int], list[str]]: Collection containing the generated or selected values.

    Example:
        >>> result = _collect_dataset_channel_info(dataset="2d_time", zarrs=Path("path/to/resource"))
    """
    arr0, axes0 = load_ome_zarr(zarrs[0], level=0, as_numpy=False)
    x0 = _to_numpy(arr0)
    x0, axes0 = _ensure_cyx(x0, axes0)
    n_c = x0.shape[axes0.index("c")] if "c" in axes0 else 1

    if dataset == "2d_time":
        return [0], ["blue"]
    if n_c == 1:
        return [0], ["blue"]
    return [0, 1], ["blue", "green"]


def _load_planes(path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load planes from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Collection containing the generated or selected values.

    Example:
        >>> result = _load_planes(path=Path("path/to/resource"), dataset="2d_time")
    """
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    blue = _extract_display_plane(x, axes, 0)
    green = None
    if dataset != "2d_time" and ("c" in axes) and x.shape[axes.index("c")] > 1:
        green = _extract_display_plane(x, axes, 1)
    return blue, green


def _write_single_metrics_report(
    rep: Path,
    dataset: str,
    stem: str,
    params: NotchParams,
    metrics_before: dict,
    metrics_after: dict,
) -> None:
    """Write single metrics report to persistent storage.

    Args:
        rep (Path): Filesystem path used for rep.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        stem (str): Text value specifying stem.
        params (NotchParams): Value specifying params for the operation.
        metrics_before (dict): Value specifying metrics before for the operation.
        metrics_after (dict): Value specifying metrics after for the operation.

    Example:
        >>> _write_single_metrics_report(
        ...     rep=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     stem="stem",
        ...     params=...,
        ...     metrics_before={},
        ...     metrics_after={},
        ... )
    """
    with rep.open("w", encoding="utf-8") as f:
        f.write("=== SETTINGS ===\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"image: {stem}\n")
        f.write(f"angles_deg: {params.angles_deg}\n")
        f.write(f"half_width_deg: {params.half_width_deg}\n")
        f.write(f"r_min: {params.r_min}\n")
        f.write(f"r_max: {params.r_max}\n")
        f.write(f"depth: {params.depth}\n")
        f.write(f"smooth: {params.smooth}\n")
        write_metrics_block(f, "ORIGINAL METRICS", metrics_before)
        write_metrics_block(f, "FILTERED METRICS", metrics_after)


def _rows_for_batch(stem: str, metrics_before: dict, metrics_after: dict) -> list[dict]:
    """Return rows for batch for the supplied inputs.

    Args:
        stem (str): Text value specifying stem.
        metrics_before (dict): Value specifying metrics before for the operation.
        metrics_after (dict): Value specifying metrics after for the operation.

    Returns:
        list[dict]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _rows_for_batch(
        ...     stem="stem",
        ...     metrics_before={},
        ...     metrics_after={},
        ... )
    """
    rows: list[dict] = []
    for ch_name in metrics_after.keys():
        b = metrics_before[ch_name]
        a = metrics_after[ch_name]
        rows.append({
            "image": stem,
            "channel": ch_name,
            "neighbor_corr_before": b["neighbor_corr"],
            "neighbor_corr_after": a["neighbor_corr"],
            "delta_neighbor_corr": a["neighbor_corr"] - b["neighbor_corr"],
            "fft_peak_score_before": b["fft_peak_score"],
            "fft_peak_score_after": a["fft_peak_score"],
            "delta_fft_peak_score": a["fft_peak_score"] - b["fft_peak_score"],
            "grad_mag_mean_before": b["grad_mag_mean"],
            "grad_mag_mean_after": a["grad_mag_mean"],
            "delta_grad_mag_mean": a["grad_mag_mean"] - b["grad_mag_mean"],
        })
    return rows


def _write_batch_summary_txt(out_txt: Path, rows: list[dict], dataset: str, params: NotchParams) -> None:
    """Write batch summary txt to persistent storage.

    Args:
        out_txt (Path): Filesystem path used for out txt.
        rows (list[dict]): Value specifying rows for the operation.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        params (NotchParams): Value specifying params for the operation.

    Example:
        >>> _write_batch_summary_txt(
        ...     out_txt=Path("path/to/resource"),
        ...     rows=[],
        ...     dataset="2d_time",
        ...     params=...,
        ... )
    """
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("=== CURATED SUBSET BATCH SUMMARY ===\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"n_rows: {len(rows)}\n")
        f.write(f"angles_deg: {params.angles_deg}\n")
        f.write(f"half_width_deg: {params.half_width_deg}\n")
        f.write(f"r_min: {params.r_min}\n")
        f.write(f"r_max: {params.r_max}\n")
        f.write(f"depth: {params.depth}\n")
        f.write(f"smooth: {params.smooth}\n")
        if rows:
            d_nc = np.mean([r["delta_neighbor_corr"] for r in rows])
            d_fp = np.mean([r["delta_fft_peak_score"] for r in rows])
            d_gm = np.mean([r["delta_grad_mag_mean"] for r in rows])
            f.write("\n=== MEAN DELTAS ===\n")
            f.write(f"neighbor_corr: {d_nc}\n")
            f.write(f"fft_peak_score: {d_fp}\n")
            f.write(f"grad_mag_mean: {d_gm}\n")
            f.write("\n=== PER IMAGE / CHANNEL ===\n")
            for r in rows:
                f.write(
                    f"{r['image']} [{r['channel']}]: "
                    f"delta_neighbor_corr={r['delta_neighbor_corr']}, "
                    f"delta_fft_peak_score={r['delta_fft_peak_score']}, "
                    f"delta_grad_mag_mean={r['delta_grad_mag_mean']}\n"
                )


def _prompt_notch_params(default_r_min: int) -> NotchParams:
    """Return prompt notch params for the supplied inputs.

    Args:
        default_r_min (int): Numerical value controlling default r min.

    Returns:
        NotchParams: Result produced by the operation.

    Example:
        >>> result = _prompt_notch_params(default_r_min=1)
    """
    print("\nEnter NOTCH parameters (press Enter for defaults):")
    angles = _prompt_floats("  angles_deg (e.g. '90 270') [default 90 270]: ", default=[90.0, 270.0])
    half_width = _prompt_float("  half_width_deg [default 5.0]: ", default=5.0)
    depth = _prompt_float("  depth (0..1) [default 1.0]: ", default=1.0)
    smooth = _prompt_bool("  smooth edges? [y/N]: ", default=False)
    r_min = _prompt_int(f"  r_min / DC radius [default {default_r_min}]: ", default=default_r_min)
    r_max_in = input("  r_max (blank=None): ").strip()
    r_max = None if not r_max_in else int(r_max_in)
    return NotchParams(
        angles_deg=angles,
        half_width_deg=half_width,
        r_min=r_min,
        r_max=r_max,
        depth=depth,
        smooth=smooth,
    )


def _save_mean_and_example_fft_overlays(
    tuning_dir: Path,
    mean_ffts: dict[str, np.ndarray],
    selected_channel: str,
    r_dc: int,
    example_path: Path,
    dataset: str,
    fft_size: int,
) -> None:
    """Save mean and example Fourier-transform result overlays to persistent storage.

    Args:
        tuning_dir (Path): Directory used for tuning.
        mean_ffts (dict[str, np.ndarray]): Array containing mean ffts.
        selected_channel (str): Text value specifying selected channel.
        r_dc (int): Numerical value controlling r dc.
        example_path (Path): Filesystem path associated with example.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        fft_size (int): Size parameter controlling Fourier-transform result.

    Example:
        >>> _save_mean_and_example_fft_overlays(
        ...     tuning_dir=Path("path/to/resource"),
        ...     mean_ffts=image_array,
        ...     selected_channel="selected_channel",
        ...     r_dc=1,
        ...     example_path=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     fft_size=1,
        ... )
    """
    for name, m in mean_ffts.items():
        _save_fft_with_dc_overlay(
            m,
            r_dc,
            tuning_dir / f"mean_fft_dc_overlay_{name}.png",
            f"Mean FFT ({name}) with DC radius r={r_dc}",
            cmap=_channel_cmap(name),
            linewidth=3,
        )
    blue, green = _load_planes(example_path, dataset)
    _save_fft_with_dc_overlay(
        _fft_logmag(_center_crop(blue, fft_size)),
        r_dc,
        tuning_dir / "chosen_image_fft_dc_overlay_blue.png",
        f"Chosen image FFT (blue) with DC radius r={r_dc}",
        cmap=_channel_cmap("blue"),
        linewidth=3,
    )
    if green is not None:
        _save_fft_with_dc_overlay(
            _fft_logmag(_center_crop(green, fft_size)),
            r_dc,
            tuning_dir / "chosen_image_fft_dc_overlay_green.png",
            f"Chosen image FFT (green) with DC radius r={r_dc}",
            cmap=_channel_cmap("green"),
            linewidth=3,
        )


def _process_one_image(dataset: str, idx: int, zarrs: list[Path], params: NotchParams, fft_size: int) -> tuple[Path, dict, dict]:
    """Process one image using the configured workflow.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        idx (int): Numerical value controlling idx.
        zarrs (list[Path]): Filesystem path used for zarrs.
        params (NotchParams): Value specifying params for the operation.
        fft_size (int): Size parameter controlling Fourier-transform result.

    Returns:
        tuple[Path, dict, dict]: Resolved or generated filesystem path.

    Example:
        >>> result = _process_one_image(
        ...     dataset="2d_time",
        ...     idx=1,
        ...     zarrs=Path("path/to/resource"),
        ...     params=...,
        ...     fft_size=1,
        ... )
    """
    orig_path = zarrs[idx]
    stem = orig_path.parent.name
    out_dir = run_notch_on_dataset(dataset, params, apply=True, channel_mode="auto", image_index=idx)
    out_zarr = out_dir / "image.ome.zarr"

    blue_before, green_before = _load_planes(orig_path, dataset)
    blue_after, green_after = _load_planes(out_zarr, dataset)

    metrics_before = compute_metrics_per_channel(blue_before, green_before)
    metrics_after = compute_metrics_per_channel(blue_after, green_after)
    _write_single_metrics_report(out_dir / "metrics.txt", dataset, stem, params, metrics_before, metrics_after)

    _plot_rgb_comparison(
        _to_rgb_from_blue_green(blue_before, green_before),
        _to_rgb_from_blue_green(blue_after, green_after),
        f"{dataset} / {stem}",
        out_dir / "compare_original_filtered_diff.png",
    )

    _save_fft_with_dc_overlay(
        _fft_logmag(_center_crop(blue_before, fft_size)),
        params.r_min,
        out_dir / "chosen_image_fft_dc_overlay_blue.png",
        f"Chosen image FFT (blue) with DC radius r={params.r_min}",
        cmap=_channel_cmap("blue"),
        linewidth=3,
    )
    if green_before is not None:
        _save_fft_with_dc_overlay(
            _fft_logmag(_center_crop(green_before, fft_size)),
            params.r_min,
            out_dir / "chosen_image_fft_dc_overlay_green.png",
            f"Chosen image FFT (green) with DC radius r={params.r_min}",
            cmap=_channel_cmap("green"),
            linewidth=3,
        )

    return out_dir, metrics_before, metrics_after


def _process_curated_subset(dataset: str, zarrs: list[Path], curated_paths: list[Path], params: NotchParams, fft_size: int) -> Path:
    """Process curated subset using the configured workflow.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        zarrs (list[Path]): Filesystem path used for zarrs.
        curated_paths (list[Path]): Filesystem path used for curated paths.
        params (NotchParams): Value specifying params for the operation.
        fft_size (int): Size parameter controlling Fourier-transform result.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _process_curated_subset(
        ...     dataset="2d_time",
        ...     zarrs=Path("path/to/resource"),
        ...     curated_paths=Path("path/to/resource"),
        ...     params=...,
        ...     fft_size=1,
        ... )
    """
    batch_rows: list[dict] = []
    for p in curated_paths:
        idx = _path_to_index(zarrs, p)
        out_dir, before, after = _process_one_image(dataset, idx, zarrs, params, fft_size)
        batch_rows.extend(_rows_for_batch(p.parent.name, before, after))

    dataset_out = results_filters_dir() / "Notch" / dataset
    _write_batch_summary_txt(dataset_out / "batch_metrics_curated_subset.txt", batch_rows, dataset, params)
    return dataset_out


def main():
    """Execute the command-line workflow and return its process exit status.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, choices=["2d_time", "2d_wga_dapi", "2d_dpa_wagi"])
    ap.add_argument("--fft_size", type=int, default=512)
    ap.add_argument("--image_index", type=int, default=None)
    args = ap.parse_args()

    dataset = args.dataset or _choose_dataset_interactive()
    if dataset == "2d_dpa_wagi":
        dataset = "2d_wga_dapi"

    zarrs = list_omezarr_images(dataset)
    if not zarrs:
        raise SystemExit(f"No OME-Zarr images found for dataset={dataset}")

    print(f"Found {len(zarrs)} images in {dataset}:")
    for i, p in enumerate(zarrs[:20]):
        print(f"  [{i:02d}] {p.parent.name}")
    if len(zarrs) > 20:
        print("  ... (showing first 20)")

    mode = _choose_mode_interactive()

    if mode in {"2", "3"}:
        curated_paths = list_curated_test_images(dataset)
        missing = missing_curated_test_stems(dataset)
        if not curated_paths:
            raise SystemExit(f"No workflow images configured/found for dataset={dataset}")

        if dataset == "2d_wga_dapi":
            print(f"Using ALL images for 2d_wga_dapi: {len(curated_paths)} images")
        else:
            if missing:
                print("Warning: some curated stems were not found:")
                for stem in missing:
                    print(f"  - {stem}")
            print(f"Curated subset size: {len(curated_paths)}")

        for p in curated_paths:
            print(f"  - {p.parent.name}")

    channels, channel_names = _collect_dataset_channel_info(dataset, zarrs)

    if mode == "1":
        idx = args.image_index
        if idx is None:
            idx = int(np.clip(_prompt_int(f"Choose image index [0..{len(zarrs)-1}] [default 0]: ", 0), 0, len(zarrs) - 1))
        params = _prompt_notch_params(default_r_min=72)
        do_apply = _prompt_bool("\nApply these parameters now? [y/N]: ", default=False)
        if not do_apply:
            print("Cancelled.")
            return
        out_dir, _, _ = _process_one_image(dataset, idx, zarrs, params, args.fft_size)
        print("Saved outputs:", out_dir)
        return

    tuning_dir = results_filters_dir() / "Notch" / dataset / "tuning_curated_subset"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    if mode == "2":
        mean_ffts: Dict[str, np.ndarray] = {}
        for ch, name in zip(channels, channel_names):
            mean_ffts[name] = mean_fft_magnitude(curated_paths, channel_index=ch, fft_size=args.fft_size)
            fig, ax = plt.subplots()
            _imshow_fft(ax, mean_ffts[name], cmap=_channel_cmap(name))
            if dataset == "2d_wga_dapi":
                ax.set_title(f"Mean FFT log-magnitude (ALL images, {dataset}) - {name}")
            else:
                ax.set_title(f"Mean FFT log-magnitude (curated subset, {dataset}) - {name}")
            ax.axis("off")
        plt.show()

        if len(channel_names) > 1:
            choice = input("Select channel for DC selection [blue/green] (default blue): ").strip().lower() or "blue"
            if choice not in mean_ffts:
                choice = "blue"
        else:
            choice = channel_names[0]

        r_dc = _pick_radius_centered(mean_ffts[choice], f"DC radius on mean FFT ({choice})", cmap=_channel_cmap(choice))
        print(f"Selected DC radius: r_dc={r_dc}px")

        for name, m in mean_ffts.items():
            th, prof = angle_energy_profile(m, r_min=r_dc)
            plt.figure()
            plt.plot(th, prof)
            if dataset == "2d_wga_dapi":
                plt.title(f"Angle-energy profile (ALL images, {dataset}) - {name} (r_dc={r_dc})")
            else:
                plt.title(f"Angle-energy profile ({dataset}) - {name} (r_dc={r_dc})")
            plt.xlabel("Angle (deg)")
            plt.ylabel("Energy (arb.)")
            plt.xlim(0, 360)
        plt.show()

        params = _prompt_notch_params(default_r_min=r_dc)
        _save_mean_and_example_fft_overlays(tuning_dir, mean_ffts, choice, params.r_min, curated_paths[0], dataset, args.fft_size)
        do_apply = _prompt_bool("\nApply these tuned parameters and run processing now? [y/N]: ", default=False)
        if not do_apply:
            print("Tuning finished without processing. Saved tuning overlays:", tuning_dir)
            return

        dataset_out = _process_curated_subset(dataset, zarrs, curated_paths, params, args.fft_size)
        print("Saved curated subset outputs:", dataset_out)
        return

    params = _prompt_notch_params(default_r_min=72)
    do_apply = _prompt_bool("\nRun processing on curated subset now? [y/N]: ", default=False)
    if not do_apply:
        print("Cancelled.")
        return
    dataset_out = _process_curated_subset(dataset, zarrs, curated_paths, params, args.fft_size)
    print("Saved curated subset outputs:", dataset_out)


if __name__ == "__main__":
    main()