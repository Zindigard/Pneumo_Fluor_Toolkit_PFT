from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import list_omezarr_images, _ensure_cyx, _to_numpy
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.free_hand_filter import (
    FreehandMaskParams,
    results_filters_dir,
    run_freehand_on_dataset,
)

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


@dataclass
class MaskSpec:
    kind: str  # "circle", "soft", "outer_cross", or "fft_rect_delete"
    radius: int
    feather: int = 15
    band_halfwidth: int = 8
    depth: float = 0.7
    fft_rects: list[tuple[int, int, int, int]] | None = None  # x, y, w, h on FFT image

    @property
    def label(self) -> str:
        if self.kind == "soft":
            base = f"soft_r{self.radius}_f{self.feather}"
        elif self.kind == "outer_cross":
            base = (
                f"outer_cross_r{self.radius}_bw{self.band_halfwidth}"
                f"_f{self.feather}_d{int(round(self.depth * 100))}"
            )
        elif self.kind == "fft_rect_delete":
            base = "fft_rect_delete"
        else:
            base = f"circle_r{self.radius}"
        if self.kind == "fft_rect_delete" and self.fft_rects:
            parts = [f"x{x}_y{y}_w{w}_h{h}" for (x, y, w, h) in self.fft_rects]
            joined = "__".join(parts)
            return f"{base}_{joined}"
        return base

    @property
    def human_title(self) -> str:
        if self.kind == "soft":
            return f"soft mask (r={self.radius}, feather={self.feather})"
        if self.kind == "outer_cross":
            return (
                "outer cross band-stop "
                f"(protect_r={self.radius}, band_half={self.band_halfwidth}, "
                f"feather={self.feather}, depth={self.depth:.2f})"
            )
        if self.kind == "fft_rect_delete":
            if self.fft_rects:
                return f"FFT rectangle delete ({len(self.fft_rects)} rects)"
            return "FFT rectangle delete"
        return f"circle mask (r={self.radius})"


# -------------------------
# Curated subset helpers
# -------------------------
def list_curated_test_images(dataset: str) -> list[Path]:
    all_imgs = list_omezarr_images(dataset)
    want = CURATED_TEST_STEMS.get(dataset, [])
    by_stem = {p.parent.name: p for p in all_imgs}
    return [by_stem[s] for s in want if s in by_stem]



def missing_curated_test_stems(dataset: str) -> list[str]:
    all_stems = {p.parent.name for p in list_omezarr_images(dataset)}
    return [s for s in CURATED_TEST_STEMS.get(dataset, []) if s not in all_stems]


# -------------------------
# Metrics
# -------------------------
def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS
    return float(np.sum(a * b) / denom)



def neighbor_corr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    vals = []
    if x.shape[1] >= 2:
        vals.append(_pearson_corr(x[:, :-1], x[:, 1:]))
    if x.shape[0] >= 2:
        vals.append(_pearson_corr(x[:-1, :], x[1:, :]))
    return float(np.mean(vals)) if vals else 0.0



def fft_peak_score(x: np.ndarray, dc_halfwidth: int = 8) -> float:
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
    x = np.asarray(x, dtype=np.float32)
    gy, gx = np.gradient(x)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))



def compute_metrics_per_channel(blue: np.ndarray, green: np.ndarray | None) -> dict:
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
    f.write(f"\n=== {title} ===\n")
    for ch_name, m in metrics.items():
        f.write(f"\n[{ch_name}]\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")


# -------------------------
# Display helpers
# -------------------------
def _norm01_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)



def _to_rgb_from_blue_green(blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    B = _norm01_percentile(blue)
    G = _norm01_percentile(green) if green is not None else np.zeros_like(B)
    R = np.zeros_like(B)
    return np.stack([R, G, B], axis=-1)


def _scale_shared_raw(orig: np.ndarray, filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = float(min(np.min(orig), np.min(filt)))
    hi = float(max(np.max(orig), np.max(filt)))
    if hi <= lo:
        z1 = np.zeros_like(orig, dtype=np.float32)
        z2 = np.zeros_like(filt, dtype=np.float32)
        return z1, z2
    return (np.clip((orig - lo) / (hi - lo), 0.0, 1.0).astype(np.float32),
            np.clip((filt - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))


def _to_rgb_from_blue_green_raw_shared(
    blue_orig: np.ndarray,
    green_orig: np.ndarray | None,
    blue_filt: np.ndarray,
    green_filt: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    b1, b2 = _scale_shared_raw(np.asarray(blue_orig, dtype=np.float32), np.asarray(blue_filt, dtype=np.float32))
    if green_orig is not None and green_filt is not None:
        g1, g2 = _scale_shared_raw(np.asarray(green_orig, dtype=np.float32), np.asarray(green_filt, dtype=np.float32))
    else:
        g1 = np.zeros_like(b1)
        g2 = np.zeros_like(b2)
    r1 = np.zeros_like(b1)
    r2 = np.zeros_like(b2)
    return np.stack([r1, g1, b1], axis=-1), np.stack([r2, g2, b2], axis=-1)



def _make_crimson_cmap():
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



def _plot_rgb_comparison(orig_rgb: np.ndarray, filt_rgb: np.ndarray, title: str, out_png: Path, *, variant_label: str):
    diff = np.abs(filt_rgb.astype(np.float32) - orig_rgb.astype(np.float32))
    diff_map = np.mean(diff, axis=-1)
    scale = float(np.percentile(diff_map, 99.7)) + 1e-12
    diff_norm = np.clip(diff_map / scale, 0.0, 1.0)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(orig_rgb)
    ax1.set_title(f"Original ({variant_label})")
    ax1.axis("off")

    ax2.imshow(filt_rgb)
    ax2.set_title(f"Filtered ({variant_label})")
    ax2.axis("off")

    ax3.imshow(diff_norm, cmap=_make_crimson_cmap(), vmin=0, vmax=1)
    ax3.set_title("|Δ| (black=min, crimson=max)")
    ax3.axis("off")

    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("black")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _save_both_comparisons(
    blue_before: np.ndarray,
    green_before: np.ndarray | None,
    blue_after: np.ndarray,
    green_after: np.ndarray | None,
    title: str,
    out_dir: Path,
) -> None:
    raw_orig_rgb, raw_filt_rgb = _to_rgb_from_blue_green_raw_shared(
        blue_before, green_before, blue_after, green_after
    )
    _plot_rgb_comparison(
        raw_orig_rgb,
        raw_filt_rgb,
        title,
        out_dir / "compare_original_filtered_diff_raw.png",
        variant_label="RGB, shared raw scale",
    )
    _plot_rgb_comparison(
        _to_rgb_from_blue_green(blue_before, green_before),
        _to_rgb_from_blue_green(blue_after, green_after),
        title,
        out_dir / "compare_original_filtered_diff_normalized.png",
        variant_label="RGB, display-norm",
    )


# -------------------------
# FFT helpers
# -------------------------
def _fft_logmag(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32, copy=False)
    x = x - float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x))
    return np.log1p(np.abs(F)).astype(np.float32)



def _center_crop(img2d: np.ndarray, target: int = 512) -> np.ndarray:
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



def _channel_cmap(name: str) -> str:
    return {"blue": "Blues", "green": "Greens"}.get(name, "gray")



def _imshow_fft(ax, fft_img: np.ndarray, cmap: str):
    vmin = float(np.percentile(fft_img, 1.0))
    vmax = float(np.percentile(fft_img, 99.7))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    ax.set_facecolor("black")
    ax.imshow(fft_img, cmap=cmap, vmin=vmin, vmax=vmax)



def _pick_radius_centered(mean_fft: np.ndarray, title: str, cmap: str) -> int:
    h, w = mean_fft.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    _imshow_fft(ax, mean_fft, cmap=cmap)
    ax.set_title(title + "\nMove mouse (red preview), click once to set radius")
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
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        r = float(np.sqrt((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2))
        circ.set_radius(r)
        overlay_img.set_alpha((rr <= r).astype(np.float32) * 0.25)
        fig.canvas.draw_idle()

    def on_click(event):
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
        raise RuntimeError("Radius selection cancelled.")
    return int(selected["r"])



def _mirror_rect_on_fft(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, rw, rh = rect
    h, w = shape
    x1 = x0 + rw - 1
    y1 = y0 + rh - 1
    mx0 = w - 1 - x1
    my0 = h - 1 - y1
    return int(mx0), int(my0), int(rw), int(rh)


def _save_fft_with_mask_overlay(
    fft_img: np.ndarray,
    mask_keep: np.ndarray,
    out_png: Path,
    title: str,
    cmap: str,
    spec: MaskSpec,
    linewidth: int = 3,
):
    h, w = fft_img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    _imshow_fft(ax, fft_img, cmap=cmap)
    ax.axis("off")
    ax.set_title(title)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.55
    overlay[..., 2] = 0.0

    if spec.kind == "outer_cross":
        denom = max(EPS, float(spec.depth))
        overlay_alpha = np.clip((1.0 - mask_keep.astype(np.float32)) / denom, 0.0, 1.0) * 0.25
    elif spec.kind == "fft_rect_delete":
        overlay_alpha = np.clip(1.0 - mask_keep.astype(np.float32), 0.0, 1.0) * 0.35
    else:
        overlay_alpha = np.clip(mask_keep.astype(np.float32), 0.0, 1.0) * 0.25
    ax.imshow(overlay, alpha=overlay_alpha)

    if spec.kind == "fft_rect_delete":
        if spec.fft_rects:
            for rect in spec.fft_rects:
                x0, y0, rw, rh = rect
                ax.add_patch(Rectangle((x0, y0), rw, rh, fill=False, edgecolor="orange", linewidth=linewidth))
                mx0, my0, mrw, mrh = _mirror_rect_on_fft(rect, (h, w))
                if (mx0, my0, mrw, mrh) != rect:
                    ax.add_patch(
                        Rectangle(
                            (mx0, my0), mrw, mrh, fill=False, edgecolor="gold",
                            linewidth=max(1, linewidth - 1), linestyle="--"
                        )
                    )
    else:
        ax.add_patch(Circle((cx, cy), spec.radius, fill=False, edgecolor="orange", linewidth=linewidth))

        if spec.kind == "outer_cross":
            if spec.feather > 0:
                ax.add_patch(
                    Circle(
                        (cx, cy),
                        spec.radius + spec.feather,
                        fill=False,
                        edgecolor="gold",
                        linewidth=max(1, linewidth - 1),
                        linestyle="--",
                    )
                )
            ax.add_patch(
                Rectangle(
                    (cx - spec.band_halfwidth, -0.5),
                    2 * spec.band_halfwidth,
                    h,
                    fill=False,
                    edgecolor="orange",
                    linewidth=max(1, linewidth - 1),
                )
            )
            ax.add_patch(
                Rectangle(
                    (-0.5, cy - spec.band_halfwidth),
                    w,
                    2 * spec.band_halfwidth,
                    fill=False,
                    edgecolor="orange",
                    linewidth=max(1, linewidth - 1),
                )
            )
            if spec.feather > 0:
                bw = spec.band_halfwidth + spec.feather
                ax.add_patch(
                    Rectangle(
                        (cx - bw, -0.5),
                        2 * bw,
                        h,
                        fill=False,
                        edgecolor="gold",
                        linewidth=max(1, linewidth - 2),
                        linestyle="--",
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (-0.5, cy - bw),
                        w,
                        2 * bw,
                        fill=False,
                        edgecolor="gold",
                        linewidth=max(1, linewidth - 2),
                        linestyle="--",
                    )
                )
        elif spec.feather > 0:
            ax.add_patch(
                Circle(
                    (cx, cy),
                    spec.radius + spec.feather,
                    fill=False,
                    edgecolor="gold",
                    linewidth=max(1, linewidth - 1),
                    linestyle="--",
                )
            )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Mask builders
# -------------------------
def _grid_radius(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return yy, xx, rr, cx, cy



def _cosine_keep_profile(distance: np.ndarray, radius: float, feather: int) -> np.ndarray:
    feather = max(1, int(feather))
    out = np.zeros_like(distance, dtype=np.float32)
    out[distance <= radius] = 1.0
    band = (distance > radius) & (distance < radius + feather)
    t = (distance[band] - radius) / float(feather)
    out[band] = 0.5 * (1.0 + np.cos(np.pi * t))
    return out



def _cosine_ramp_profile(distance: np.ndarray, start: float, feather: int) -> np.ndarray:
    feather = max(1, int(feather))
    out = np.ones_like(distance, dtype=np.float32)
    out[distance <= start] = 0.0
    band = (distance > start) & (distance < start + feather)
    t = (distance[band] - start) / float(feather)
    out[band] = 0.5 * (1.0 - np.cos(np.pi * t))
    return out



def _build_center_keep_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    _, _, rr, _, _ = _grid_radius(shape)
    return (rr <= float(radius)).astype(np.float32)



def _build_soft_center_keep_mask(shape: tuple[int, int], radius: int, feather: int) -> np.ndarray:
    """
    Soft center-keep mask.

    - fully keep frequencies for r <= radius
    - cosine-taper from radius .. radius+feather
    - fully suppress for r >= radius+feather
    """
    _, _, rr, _, _ = _grid_radius(shape)
    return _cosine_keep_profile(rr, float(radius), feather)



def _build_outer_cross_soft_mask(
    shape: tuple[int, int],
    protect_radius: int,
    band_halfwidth: int,
    feather: int,
    depth: float,
) -> np.ndarray:
    """
    Targeted outer cross band-stop.

    - protected center stays untouched inside protect_radius
    - only frequencies near the vertical/horizontal axes are attenuated
      outside the protected center
    - attenuation is soft both radially and across the band width
    - depth controls maximum attenuation strength
    """
    feather = max(1, int(feather))
    band_halfwidth = max(1, int(band_halfwidth))
    depth = float(np.clip(depth, 0.0, 1.0))

    yy, xx, rr, cx, cy = _grid_radius(shape)
    dv = np.abs(xx - cx).astype(np.float32)
    dh = np.abs(yy - cy).astype(np.float32)

    radial_gate = _cosine_ramp_profile(rr, float(protect_radius), feather)
    vertical_profile = _cosine_keep_profile(dv, float(band_halfwidth), feather)
    horizontal_profile = _cosine_keep_profile(dh, float(band_halfwidth), feather)

    suppress = np.maximum(vertical_profile, horizontal_profile) * radial_gate
    mask_keep = 1.0 - depth * suppress
    return np.clip(mask_keep.astype(np.float32), 0.0, 1.0)



def _build_fft_rect_delete_mask(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.ones(shape, dtype=np.float32)
    h, w = shape
    for rect in rects:
        x0, y0, rw, rh = rect
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        rw = int(np.clip(rw, 1, w - x0))
        rh = int(np.clip(rh, 1, h - y0))
        mask[y0:y0 + rh, x0:x0 + rw] = 0.0
        mx0, my0, mrw, mrh = _mirror_rect_on_fft((x0, y0, rw, rh), shape)
        mx0 = int(np.clip(mx0, 0, w - 1))
        my0 = int(np.clip(my0, 0, h - 1))
        mrw = int(np.clip(mrw, 1, w - mx0))
        mrh = int(np.clip(mrh, 1, h - my0))
        mask[my0:my0 + mrh, mx0:mx0 + mrw] = 0.0
    return mask


def _pick_rectangle_on_fft(fft_img: np.ndarray, title: str, cmap: str) -> tuple[int, int, int, int]:
    fig, ax = plt.subplots()
    _imshow_fft(ax, fft_img, cmap=cmap)
    ax.set_title(title + "\nDrag rectangle, press Enter to accept, Esc to cancel")
    ax.axis("off")

    selected = {"rect": None}
    h, w = fft_img.shape

    def _clip_rect(x1, y1, x2, y2):
        xa = int(np.floor(min(x1, x2)))
        xb = int(np.ceil(max(x1, x2)))
        ya = int(np.floor(min(y1, y2)))
        yb = int(np.ceil(max(y1, y2)))
        xa = int(np.clip(xa, 0, w - 1))
        ya = int(np.clip(ya, 0, h - 1))
        xb = int(np.clip(xb, xa + 1, w))
        yb = int(np.clip(yb, ya + 1, h))
        return xa, ya, xb - xa, yb - ya

    def on_select(eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        selected["rect"] = _clip_rect(eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    rs = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
        drag_from_anywhere=True,
    )

    def on_key(event):
        if event.key == "enter":
            if selected["rect"] is None:
                x1, x2, y1, y2 = rs.extents
                if abs(x2 - x1) >= 1 and abs(y2 - y1) >= 1:
                    selected["rect"] = _clip_rect(x1, y1, x2, y2)
            plt.close(fig)
        elif event.key == "escape":
            selected["rect"] = None
            plt.close(fig)

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    rs.set_active(False)

    if selected["rect"] is None:
        raise RuntimeError("FFT rectangle selection cancelled.")
    return tuple(int(v) for v in selected["rect"])


def _pick_multiple_rectangles_on_fft(fft_img: np.ndarray, title: str, cmap: str) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    while True:
        suffix = f" #{len(rects) + 1}"
        rect = _pick_rectangle_on_fft(fft_img, title + suffix, cmap)
        rects.append(tuple(int(v) for v in rect))
        more = _prompt_bool("Add another FFT delete rectangle? [y/N]: ", default=False)
        if not more:
            break
    return rects


def _build_mask(shape: tuple[int, int], spec: MaskSpec) -> np.ndarray:
    if spec.kind == "soft":
        return _build_soft_center_keep_mask(shape, spec.radius, spec.feather)
    if spec.kind == "outer_cross":
        return _build_outer_cross_soft_mask(
            shape,
            protect_radius=spec.radius,
            band_halfwidth=spec.band_halfwidth,
            feather=spec.feather,
            depth=spec.depth,
        )
    if spec.kind == "fft_rect_delete":
        if not spec.fft_rects:
            raise ValueError("fft_rect_delete requires spec.fft_rects")
        return _build_fft_rect_delete_mask(shape, spec.fft_rects)
    return _build_center_keep_mask(shape, spec.radius)


def _apply_frequency_mask_one_plane(img2d: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32, copy=False)
    mu = float(np.mean(x))
    x0 = x - mu
    F = np.fft.fftshift(np.fft.fft2(x0))
    if mask_keep.shape != x.shape:
        raise ValueError(f"mask_keep shape {mask_keep.shape} != image shape {x.shape}")
    Ff = F * mask_keep.astype(np.float32, copy=False)
    y = np.fft.ifft2(np.fft.ifftshift(Ff))
    return np.real(y).astype(np.float32) + mu


def _pick_filter_channels(dataset: str, axes: str, arr: np.ndarray) -> list[int]:
    n_c = arr.shape[axes.index("c")] if "c" in axes else 1
    if dataset.strip().lower() == "2d_time":
        return [0]
    return list(range(min(n_c, 2))) if n_c > 1 else [0]


# -------------------------
# CLI helpers
# -------------------------
def _choose_dataset_interactive() -> str:
    print("Choose dataset:")
    print("  1) 2d_time (blue only)")
    print("  2) 2d_wga_dapi (blue + green)")
    ans = input("Choose number [1/2]: ").strip() or "1"
    return "2d_wga_dapi" if ans == "2" else "2d_time"



def _choose_mode_interactive() -> str:
    print("Choose mode:")
    print("  1) Process ONE image directly (enter mask parameters first, skip tuning figures)")
    print("  2) Tune on curated subset first, then process curated subset")
    print("  3) Process curated subset directly (enter mask parameters first, skip tuning figures)")
    ans = input("Choose number [1/2/3]: ").strip() or "2"
    return ans if ans in {"1", "2", "3"} else "2"



def _choose_mask_kind_interactive(allow_fft_rect_delete: bool = False) -> str:
    print("Choose free-hand mask type:")
    print("  1) Hard center circle (current logic)")
    print("  2) Soft / feathered center circle")
    print("  3) Targeted outer cross band-stop (soft, outside protected center)")
    if allow_fft_rect_delete:
        print("  4) FFT rectangle delete (ONE image only)")
    prompt = "Choose number [1/2/3/4]: " if allow_fft_rect_delete else "Choose number [1/2/3]: "
    ans = input(prompt).strip() or "1"
    if ans == "2":
        return "soft"
    if ans == "3":
        return "outer_cross"
    if allow_fft_rect_delete and ans == "4":
        return "fft_rect_delete"
    return "circle"



def _prompt_bool(prompt: str, default: bool = False) -> bool:
    s = input(prompt).strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true", "t")



def _prompt_int(prompt: str, default: int) -> int:
    s = input(prompt).strip()
    return default if not s else int(s)



def _prompt_float(prompt: str, default: float) -> float:
    s = input(prompt).strip()
    return default if not s else float(s)



def _prompt_mask_spec_interactive(
    default_radius: int = 72,
    default_feather: int = 15,
    default_band_halfwidth: int = 8,
    default_depth: float = 0.7,
    *,
    allow_fft_rect_delete: bool = False,
) -> MaskSpec:
    kind = _choose_mask_kind_interactive(allow_fft_rect_delete=allow_fft_rect_delete)
    if kind == "fft_rect_delete":
        return MaskSpec(kind="fft_rect_delete", radius=0, feather=0)
    if kind == "outer_cross":
        radius = _prompt_int(
            f"Enter protected center radius in pixels [default {default_radius}]: ",
            default_radius,
        )
        band_halfwidth = _prompt_int(
            f"Enter outer cross band half-width in pixels [default {default_band_halfwidth}]: ",
            default_band_halfwidth,
        )
        feather = _prompt_int(
            f"Enter feather width in pixels [default {default_feather}]: ",
            default_feather,
        )
        depth = _prompt_float(
            f"Enter attenuation depth 0..1 [default {default_depth:.2f}]: ",
            default_depth,
        )
        return MaskSpec(
            kind="outer_cross",
            radius=radius,
            feather=max(1, feather),
            band_halfwidth=max(1, band_halfwidth),
            depth=float(np.clip(depth, 0.0, 1.0)),
        )

    radius = _prompt_int(f"Enter center-keep radius in pixels [default {default_radius}]: ", default_radius)
    if kind == "soft":
        feather = _prompt_int(f"Enter feather width in pixels [default {default_feather}]: ", default_feather)
        feather = max(1, feather)
        return MaskSpec(kind="soft", radius=radius, feather=feather)
    return MaskSpec(kind="circle", radius=radius, feather=0)



def _prompt_mask_spec_for_selected_radius(
    selected_radius: int,
    default_feather: int = 15,
    default_band_halfwidth: int = 8,
    default_depth: float = 0.7,
) -> MaskSpec:
    kind = _choose_mask_kind_interactive(allow_fft_rect_delete=False)
    if kind == "outer_cross":
        band_halfwidth = _prompt_int(
            f"Enter outer cross band half-width in pixels [default {default_band_halfwidth}]: ",
            default_band_halfwidth,
        )
        feather = _prompt_int(
            f"Enter feather width in pixels [default {default_feather}]: ",
            default_feather,
        )
        depth = _prompt_float(
            f"Enter attenuation depth 0..1 [default {default_depth:.2f}]: ",
            default_depth,
        )
        return MaskSpec(
            kind="outer_cross",
            radius=selected_radius,
            feather=max(1, feather),
            band_halfwidth=max(1, band_halfwidth),
            depth=float(np.clip(depth, 0.0, 1.0)),
        )
    if kind == "soft":
        feather = _prompt_int(f"Enter feather width in pixels [default {default_feather}]: ", default_feather)
        return MaskSpec(kind="soft", radius=selected_radius, feather=max(1, feather))
    return MaskSpec(kind="circle", radius=selected_radius, feather=0)



def _path_to_index(all_paths: list[Path], target: Path) -> int:
    target_resolved = target.resolve()
    for i, p in enumerate(all_paths):
        if p.resolve() == target_resolved:
            return i
    raise ValueError(f"Path not found in dataset list: {target}")



def _extract_display_plane(x: np.ndarray, axes: str, c: int) -> np.ndarray:
    arr = x
    if "c" in axes:
        arr = np.take(arr, indices=c, axis=axes.index("c"))
    if "t" in axes and arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D display plane, got shape={arr.shape}, axes={axes}")
    return np.asarray(arr, dtype=np.float32)



def _collect_dataset_channel_info(dataset: str, zarrs: list[Path]) -> tuple[list[int], list[str]]:
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
    arr, axes = load_ome_zarr(path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    blue = _extract_display_plane(x, axes, 0)
    green = None
    if dataset != "2d_time" and ("c" in axes) and x.shape[axes.index("c")] > 1:
        green = _extract_display_plane(x, axes, 1)
    return blue, green



def _write_single_metrics_report(rep: Path, dataset: str, stem: str, spec: MaskSpec, metrics_before: dict, metrics_after: dict) -> None:
    with rep.open("w", encoding="utf-8") as f:
        f.write("=== SETTINGS ===\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"image: {stem}\n")
        f.write(f"mask_kind: {spec.kind}\n")
        f.write(f"radius_pixels: {spec.radius}\n")
        if spec.kind in {"soft", "outer_cross"}:
            f.write(f"feather_pixels: {spec.feather}\n")
        if spec.kind == "outer_cross":
            f.write(f"band_halfwidth_pixels: {spec.band_halfwidth}\n")
            f.write(f"depth: {spec.depth}\n")
        if spec.kind == "fft_rect_delete" and spec.fft_rects:
            for i, (x, y, w, h) in enumerate(spec.fft_rects, start=1):
                f.write(f"fft_rect_{i}: x={x}, y={y}, w={w}, h={h}\n")
        write_metrics_block(f, "ORIGINAL METRICS", metrics_before)
        write_metrics_block(f, "FILTERED METRICS", metrics_after)


def _rows_for_batch(stem: str, spec: MaskSpec, metrics_before: dict, metrics_after: dict) -> list[dict]:
    rows: list[dict] = []
    for ch_name in metrics_after.keys():
        b = metrics_before[ch_name]
        a = metrics_after[ch_name]
        rows.append({
            "image": stem,
            "channel": ch_name,
            "mask_kind": spec.kind,
            "radius": spec.radius,
            "feather": spec.feather if spec.kind in {"soft", "outer_cross"} else 0,
            "band_halfwidth": spec.band_halfwidth if spec.kind == "outer_cross" else 0,
            "depth": spec.depth if spec.kind == "outer_cross" else 0.0,
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



def _write_batch_summary_txt(out_txt: Path, rows: list[dict], dataset: str, spec: MaskSpec) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("=== CURATED SUBSET BATCH SUMMARY ===\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"mask_kind: {spec.kind}\n")
        f.write(f"radius_pixels: {spec.radius}\n")
        if spec.kind in {"soft", "outer_cross"}:
            f.write(f"feather_pixels: {spec.feather}\n")
        if spec.kind == "outer_cross":
            f.write(f"band_halfwidth_pixels: {spec.band_halfwidth}\n")
            f.write(f"depth: {spec.depth}\n")
        if spec.kind == "fft_rect_delete" and spec.fft_rects:
            for i, (x, y, w, h) in enumerate(spec.fft_rects, start=1):
                f.write(f"fft_rect_{i}: x={x}, y={y}, w={w}, h={h}\n")
        f.write(f"n_rows: {len(rows)}\n")
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


def _save_mean_and_example_fft_overlays(
    tuning_dir: Path,
    mean_ffts: dict[str, np.ndarray],
    spec: MaskSpec,
    example_path: Path,
    dataset: str,
    fft_size: int,
) -> None:
    blue_example, green_example = _load_planes(example_path, dataset)
    blue_mask = _build_mask(_center_crop(blue_example, fft_size).shape, spec)
    green_mask = _build_mask(_center_crop(green_example, fft_size).shape, spec) if green_example is not None else None

    for name, m in mean_ffts.items():
        mask_keep = _build_mask(m.shape, spec)
        _save_fft_with_mask_overlay(
            m,
            mask_keep,
            tuning_dir / f"mean_fft_overlay_{name}_{spec.label}.png",
            f"Mean FFT ({name}) with {spec.human_title}",
            cmap=_channel_cmap(name),
            spec=spec,
            linewidth=3,
        )

    _save_fft_with_mask_overlay(
        _fft_logmag(_center_crop(blue_example, fft_size)),
        blue_mask,
        tuning_dir / f"chosen_image_fft_overlay_blue_{spec.label}.png",
        f"Chosen image FFT (blue) with {spec.human_title}",
        cmap=_channel_cmap("blue"),
        spec=spec,
        linewidth=3,
    )
    if green_example is not None and green_mask is not None:
        _save_fft_with_mask_overlay(
            _fft_logmag(_center_crop(green_example, fft_size)),
            green_mask,
            tuning_dir / f"chosen_image_fft_overlay_green_{spec.label}.png",
            f"Chosen image FFT (green) with {spec.human_title}",
            cmap=_channel_cmap("green"),
            spec=spec,
            linewidth=3,
        )



def _process_one_image(dataset: str, idx: int, zarrs: list[Path], spec: MaskSpec, fft_size: int) -> tuple[Path, dict, dict]:
    orig_path = zarrs[idx]
    stem = orig_path.parent.name
    blue_before, _green_before = _load_planes(orig_path, dataset)
    mask_keep = _build_mask(blue_before.shape, spec)

    out_dir = run_freehand_on_dataset(
        dataset,
        FreehandMaskParams(mask_keep=mask_keep),
        apply=True,
        channel_mode="auto",
        image_index=idx,
        out_subdir_name=spec.label,
    )
    out_zarr = out_dir / "image.ome.zarr"

    blue_before, green_before = _load_planes(orig_path, dataset)
    blue_after, green_after = _load_planes(out_zarr, dataset)

    metrics_before = compute_metrics_per_channel(blue_before, green_before)
    metrics_after = compute_metrics_per_channel(blue_after, green_after)
    _write_single_metrics_report(out_dir / "metrics.txt", dataset, stem, spec, metrics_before, metrics_after)

    _save_both_comparisons(
        blue_before,
        green_before,
        blue_after,
        green_after,
        f"{dataset} / {stem} / {spec.human_title}",
        out_dir,
    )

    _save_fft_with_mask_overlay(
        _fft_logmag(_center_crop(blue_before, fft_size)),
        _build_mask(_center_crop(blue_before, fft_size).shape, spec),
        out_dir / "chosen_image_fft_overlay_blue.png",
        f"Chosen image FFT (blue) with {spec.human_title}",
        cmap=_channel_cmap("blue"),
        spec=spec,
        linewidth=3,
    )
    if green_before is not None:
        _save_fft_with_mask_overlay(
            _fft_logmag(_center_crop(green_before, fft_size)),
            _build_mask(_center_crop(green_before, fft_size).shape, spec),
            out_dir / "chosen_image_fft_overlay_green.png",
            f"Chosen image FFT (green) with {spec.human_title}",
            cmap=_channel_cmap("green"),
            spec=spec,
            linewidth=3,
        )

    return out_dir, metrics_before, metrics_after



def _process_one_image_fft_rect_delete(dataset: str, idx: int, zarrs: list[Path]) -> tuple[Path, dict, dict]:
    in_path = zarrs[idx]
    stem = in_path.parent.name

    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    y = x.astype(np.float32, copy=True)

    blue_before = _extract_display_plane(x, axes, 0)
    fft_blue = _fft_logmag(blue_before)
    rects = _pick_multiple_rectangles_on_fft(fft_blue, "Chosen image FFT: draw rectangle to delete, then press Enter", cmap=_channel_cmap("blue"))
    spec = MaskSpec(kind="fft_rect_delete", radius=0, feather=0, fft_rects=rects)
    for i, rect in enumerate(rects, start=1):
        print(f"Selected FFT rectangle {i}: x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}")

    do_apply = _prompt_bool("\nApply this FFT rectangle delete now? [y/N]: ", default=False)
    if not do_apply:
        raise RuntimeError("Cancelled.")

    mask_keep = _build_mask(blue_before.shape, spec)
    chs = _pick_filter_channels(dataset, axes, x)
    for c in chs:
        sl = [slice(None)] * y.ndim
        if "c" in axes:
            sl[axes.index("c")] = c
        plane = y[tuple(sl)]
        if plane.ndim == 2:
            y[tuple(sl)] = _apply_frequency_mask_one_plane(plane, mask_keep)
        elif plane.ndim == 3 and "t" in axes:
            out_plane = plane.astype(np.float32, copy=True)
            for t in range(plane.shape[0]):
                out_plane[t] = _apply_frequency_mask_one_plane(plane[t], mask_keep)
            y[tuple(sl)] = out_plane
        else:
            raise ValueError(f"Unexpected plane ndim={plane.ndim} for axes={axes}")

    out_root = results_filters_dir() / "Free_hand" / dataset / spec.label
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_c = x.shape[axes.index("c")] if "c" in axes else 1
    meta = SimpleNamespace(
        pixel_size_um_x=1.0,
        pixel_size_um_y=1.0,
        pixel_size_um_z=1.0,
        channel_names=["ch0", "ch1", "ch2"][:n_c],
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

    out_zarr = out_dir / "image.ome.zarr"
    blue_before, green_before = _load_planes(in_path, dataset)
    blue_after, green_after = _load_planes(out_zarr, dataset)

    metrics_before = compute_metrics_per_channel(blue_before, green_before)
    metrics_after = compute_metrics_per_channel(blue_after, green_after)
    _write_single_metrics_report(out_dir / "metrics.txt", dataset, stem, spec, metrics_before, metrics_after)

    _save_both_comparisons(
        blue_before,
        green_before,
        blue_after,
        green_after,
        f"{dataset} / {stem} / {spec.human_title}",
        out_dir,
    )

    _save_fft_with_mask_overlay(
        fft_blue,
        mask_keep,
        out_dir / "chosen_image_fft_overlay_blue.png",
        f"Chosen image FFT (blue) with {spec.human_title}",
        cmap=_channel_cmap("blue"),
        spec=spec,
        linewidth=3,
    )
    if green_before is not None:
        _save_fft_with_mask_overlay(
            _fft_logmag(green_before),
            mask_keep,
            out_dir / "chosen_image_fft_overlay_green.png",
            f"Chosen image FFT (green) with {spec.human_title}",
            cmap=_channel_cmap("green"),
            spec=spec,
            linewidth=3,
        )

    return out_dir, metrics_before, metrics_after


def _process_curated_subset(dataset: str, zarrs: list[Path], curated_paths: list[Path], spec: MaskSpec, fft_size: int) -> Path:
    batch_rows: list[dict] = []
    for p in curated_paths:
        idx = _path_to_index(zarrs, p)
        out_dir, before, after = _process_one_image(dataset, idx, zarrs, spec, fft_size)
        batch_rows.extend(_rows_for_batch(p.parent.name, spec, before, after))

    dataset_out = results_filters_dir() / "Free_hand" / dataset / spec.label
    _write_batch_summary_txt(dataset_out / "batch_metrics_curated_subset.txt", batch_rows, dataset, spec)
    return dataset_out



def main():
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
            raise SystemExit(f"No curated subset configured/found for dataset={dataset}")
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
        spec = _prompt_mask_spec_interactive(default_radius=72, default_feather=15, allow_fft_rect_delete=True)
        print(f"Selected: {spec.human_title}")
        if spec.kind == "fft_rect_delete":
            try:
                out_dir, _, _ = _process_one_image_fft_rect_delete(dataset, idx, zarrs)
            except RuntimeError as e:
                print(str(e))
                return
        else:
            do_apply = _prompt_bool("\nApply this mask now? [y/N]: ", default=False)
            if not do_apply:
                print("Cancelled.")
                return
            out_dir, _, _ = _process_one_image(dataset, idx, zarrs, spec, args.fft_size)
        print("Saved outputs:", out_dir)
        return

    tuning_dir = results_filters_dir() / "Free_hand" / dataset / "tuning_curated_subset"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    if mode == "2":
        mean_ffts: Dict[str, np.ndarray] = {}
        for ch, name in zip(channels, channel_names):
            mean_ffts[name] = mean_fft_magnitude(curated_paths, channel_index=ch, fft_size=args.fft_size)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            _imshow_fft(ax, mean_ffts[name], cmap=_channel_cmap(name))
            ax.set_title(f"Mean FFT log-magnitude (curated subset, {dataset}) - {name}")
            ax.axis("off")
        plt.show()

        if len(channel_names) > 1:
            choice = input("Select channel for circle selection [blue/green] (default blue): ").strip().lower() or "blue"
            if choice not in mean_ffts:
                choice = "blue"
        else:
            choice = channel_names[0]

        radius = _pick_radius_centered(mean_ffts[choice], f"Protected center radius on mean FFT ({choice})", cmap=_channel_cmap(choice))
        print(f"Selected protected radius: r={radius}px")
        spec = _prompt_mask_spec_for_selected_radius(
            radius,
            default_feather=15,
            default_band_halfwidth=8,
            default_depth=0.7,
        )
        print(f"Selected: {spec.human_title}")

        _save_mean_and_example_fft_overlays(tuning_dir, mean_ffts, spec, curated_paths[0], dataset, args.fft_size)
        do_apply = _prompt_bool("\nApply this tuned mask and run processing now? [y/N]: ", default=False)
        if not do_apply:
            print("Tuning finished without processing. Saved tuning overlays:", tuning_dir)
            return

        dataset_out = _process_curated_subset(dataset, zarrs, curated_paths, spec, args.fft_size)
        print("Saved curated subset outputs:", dataset_out)
        return

    # mode == 3
    spec = _prompt_mask_spec_interactive(default_radius=72, default_feather=15)
    print(f"Selected: {spec.human_title}")
    do_apply = _prompt_bool("\nRun processing on curated subset now? [y/N]: ", default=False)
    if not do_apply:
        print("Cancelled.")
        return
    dataset_out = _process_curated_subset(dataset, zarrs, curated_paths, spec, args.fft_size)
    print("Saved curated subset outputs:", dataset_out)


if __name__ == "__main__":
    main()
