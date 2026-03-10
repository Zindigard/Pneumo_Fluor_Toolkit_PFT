from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import list_omezarr_images, results_filters_dir
from PFT.core_prog_parts.notch_filter import _to_numpy, _ensure_cyx
from PFT.core_prog_parts.free_hand_filter import FreehandMaskParams, run_freehand_on_dataset

EPS = 1e-12



# Metrics
def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS
    return float(np.sum(a * b) / denom)


def neighbor_corr(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"neighbor_corr expects 2D array, got shape={x.shape}")
    vals = []
    if x.shape[1] >= 2:
        vals.append(_pearson_corr(x[:, :-1], x[:, 1:]))
    if x.shape[0] >= 2:
        vals.append(_pearson_corr(x[:-1, :], x[1:, :]))
    return float(np.mean(vals)) if vals else 0.0


def fft_peak_score(x: np.ndarray, dc_halfwidth: int = 8) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"fft_peak_score expects 2D array, got shape={x.shape}")
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
    if x.ndim != 2:
        raise ValueError(f"gradient_mag_mean expects 2D array, got shape={x.shape}")
    gy, gx = np.gradient(x)
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g))


def compute_metrics_per_channel(blue: np.ndarray, green: np.ndarray | None) -> dict:
    """
    Compute metrics for 1-channel (blue) or 2-channel (blue+green) chosen image.
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
    f.write(f"\n=== {title} ===\n")
    for ch_name, m in metrics.items():
        f.write(f"\n[{ch_name}]\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")


# Display 
def _norm01_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _to_rgb_from_blue_green(blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    B = _norm01_percentile(blue)
    G = _norm01_percentile(green) if green is not None else np.zeros_like(B)
    R = np.zeros_like(B)
    return np.stack([R, G, B], axis=-1)


def _make_crimson_cmap():
    return LinearSegmentedColormap.from_list(
        "white_to_crimson", [(1, 1, 1), (0.86, 0.08, 0.24)]
    )


def _plot_rgb_comparison(orig_rgb: np.ndarray, filt_rgb: np.ndarray, title: str, out_png: Path):
    diff = np.abs(filt_rgb.astype(np.float32) - orig_rgb.astype(np.float32))
    diff_map = np.mean(diff, axis=-1)
    diff_norm = diff_map / (float(np.percentile(diff_map, 99.5)) + 1e-12)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(orig_rgb); ax1.set_title("Original (RGB, display-norm)"); ax1.axis("off")
    ax2.imshow(filt_rgb); ax2.set_title("Filtered (RGB, display-norm)"); ax2.axis("off")
    ax3.imshow(np.clip(diff_norm, 0, 1), cmap=_make_crimson_cmap(), vmin=0, vmax=1)
    ax3.set_title("|Δ| (crimson=more)"); ax3.axis("off")

    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# FFT compute 
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
    max_images: int = 30,
    fft_size: int = 512,
) -> np.ndarray:
    n = min(len(zarr_paths), max_images)
    if n == 0:
        raise ValueError("No images to compute mean FFT.")

    acc = None
    used = 0
    for p in zarr_paths[:n]:
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
    ax.imshow(fft_img, cmap=cmap, vmin=vmin, vmax=vmax)


def _pick_radius_centered(mean_fft: np.ndarray, title: str, cmap: str) -> int:
    """
    One-click selection from FFT center:
      - circle center fixed at FFT center
      - click once to set radius
    """
    h, w = mean_fft.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    fig, ax = plt.subplots()
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


def _save_fft_with_overlay(fft_img: np.ndarray, radius: int, out_png: Path, title: str, cmap: str, linewidth: int = 3):
    h, w = fft_img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    fig, ax = plt.subplots()
    _imshow_fft(ax, fft_img, cmap=cmap)
    ax.axis("off")
    ax.set_title(title)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.55
    overlay[..., 2] = 0.0
    ax.imshow(overlay, alpha=(rr <= float(radius)).astype(np.float32) * 0.25)

    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor="orange", linewidth=linewidth))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _choose_dataset_interactive() -> str:
    print("Choose dataset:")
    print("  1) 2d_time (blue only)")
    print("  2) 2d_wga_dapi (blue + green)")
    ans = input("Choose number [1/2]: ").strip() or "1"
    return "2d_wga_dapi" if ans == "2" else "2d_time"


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    s = input(prompt).strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true", "t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, choices=["2d_time", "2d_wga_dapi", "2d_dpa_wagi"])
    ap.add_argument("--max_images", type=int, default=30)
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

    arr0, axes0 = load_ome_zarr(zarrs[0], level=0, as_numpy=False)
    x0 = _to_numpy(arr0)
    x0, axes0 = _ensure_cyx(x0, axes0)
    n_c = x0.shape[axes0.index("c")] if "c" in axes0 else 1

    if dataset == "2d_time":
        channels = [0]
        channel_names = ["blue"]
    else:
        channels = [0] if n_c == 1 else [0, 1]
        channel_names = ["blue"] if n_c == 1 else ["blue", "green"]

    # mean FFTs first 
    mean_ffts: Dict[str, np.ndarray] = {}
    for ch, name in zip(channels, channel_names):
        mean_ffts[name] = mean_fft_magnitude(
            zarrs, channel_index=ch, max_images=args.max_images, fft_size=args.fft_size
        )
        fig, ax = plt.subplots()
        _imshow_fft(ax, mean_ffts[name], cmap=_channel_cmap(name))
        ax.set_title(f"Mean FFT log-magnitude ({dataset}) - {name}")
        ax.axis("off")
    plt.show()

    # choose which channel's mean FFT 
    if len(channel_names) > 1:
        choice = input("Select channel for circle selection [blue/green] (default blue): ").strip().lower() or "blue"
        if choice not in mean_ffts:
            choice = "blue"
    else:
        choice = channel_names[0]

    radius = _pick_radius_centered(mean_ffts[choice], f"Free-hand circle on mean FFT ({choice})", cmap=_channel_cmap(choice))
    print(f"Selected radius: r={radius}px")

    # keep-mask
    h, w = mean_ffts[choice].shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask_keep = (rr <= float(radius)).astype(np.float32)

    out_dir_base = results_filters_dir() / "Free_hand" / dataset
    out_dir_base.mkdir(parents=True, exist_ok=True)

    np.save(out_dir_base / f"mask_keep_centered_{choice}.npy", mask_keep.astype(np.float32))
    if "blue" in mean_ffts:
        _save_fft_with_overlay(mean_ffts["blue"], radius, out_dir_base / "mask_preview_mean_fft_blue.png",
                               f"Mask on MEAN FFT (blue) r={radius}", cmap=_channel_cmap("blue"), linewidth=3)
    if "green" in mean_ffts:
        _save_fft_with_overlay(mean_ffts["green"], radius, out_dir_base / "mask_preview_mean_fft_green.png",
                               f"Mask on MEAN FFT (green) r={radius}", cmap=_channel_cmap("green"), linewidth=3)

    # choose image
    if args.image_index is None:
        idx = int(input(f"Choose image index [0..{len(zarrs)-1}]: ").strip() or "0")
    else:
        idx = int(args.image_index)
    idx = int(np.clip(idx, 0, len(zarrs) - 1))
    test_path = zarrs[idx]
    stem = test_path.parent.name
    print(f"Selected: {stem}")

    # load chosen image
    arr, axes = load_ome_zarr(test_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)

    def get_plane(ci: int) -> np.ndarray:
        if "c" in axes:
            plane = np.take(x, indices=ci, axis=axes.index("c"))
        else:
            plane = x
        if "t" in axes and plane.ndim == 3:
            plane = plane[0]
        return plane.astype(np.float32)

    blue = get_plane(0)
    green = get_plane(1) if ("c" in axes and x.shape[axes.index("c")] > 1) and dataset != "2d_time" else None

    out_img_dir = out_dir_base / stem
    out_img_dir.mkdir(parents=True, exist_ok=True)

  
    fft_blue = _fft_logmag(_center_crop(blue, args.fft_size))
    _save_fft_with_overlay(fft_blue, radius, out_img_dir / "circle_on_fft_original_blue.png",
                           f"Circle on FFT of ORIGINAL (blue) r={radius}", cmap=_channel_cmap("blue"), linewidth=3)
    if green is not None:
        fft_green = _fft_logmag(_center_crop(green, args.fft_size))
        _save_fft_with_overlay(fft_green, radius, out_img_dir / "circle_on_fft_original_green.png",
                               f"Circle on FFT of ORIGINAL (green) r={radius}", cmap=_channel_cmap("green"), linewidth=3)

    # metrics 
    rep = out_img_dir / "metrics.txt"
    orig_metrics = compute_metrics_per_channel(blue, green)
    with rep.open("w", encoding="utf-8") as f:
        f.write("=== SETTINGS ===\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"image: {stem}\n")
        f.write(f"radius_pixels: {radius}\n")
        f.write(f"mask_saved: {out_dir_base / f'mask_keep_centered_{choice}.npy'}\n")
        write_metrics_block(f, "ORIGINAL METRICS (chosen image)", orig_metrics)

    orig_rgb = _to_rgb_from_blue_green(blue, green)

    do_apply = _prompt_bool("Apply free-hand filter now? [y/N]: ", default=False)
    if not do_apply:
        _plot_rgb_comparison(orig_rgb, orig_rgb.copy(), f"{dataset} / {stem} (DRY-RUN)", out_img_dir / "compare_original_filtered_diff.png")
        print("Dry-run (no OME-Zarr saved). Saved:", out_img_dir)
        return

    params = FreehandMaskParams(mask_keep=mask_keep.astype(np.float32))
    out_applied_dir = run_freehand_on_dataset(dataset, params, apply=True, channel_mode="auto", image_index=idx)
    print("Saved filtered OME-Zarr:", out_applied_dir / "image.ome.zarr")

    arrf, axesf = load_ome_zarr(out_applied_dir / "image.ome.zarr", level=0, as_numpy=False)
    yf = _to_numpy(arrf)
    yf, axesf = _ensure_cyx(yf, axesf)

    def get_plane_f(ci: int) -> np.ndarray:
        if "c" in axesf:
            plane = np.take(yf, indices=ci, axis=axesf.index("c"))
        else:
            plane = yf
        if "t" in axesf and plane.ndim == 3:
            plane = plane[0]
        return plane.astype(np.float32)

    blue_f = get_plane_f(0)
    green_f = get_plane_f(1) if ("c" in axesf and yf.shape[axesf.index("c")] > 1) and dataset != "2d_time" else None

    filt_metrics = compute_metrics_per_channel(blue_f, green_f)
    with rep.open("a", encoding="utf-8") as f:
        write_metrics_block(f, "FILTERED METRICS (chosen image)", filt_metrics)

    filt_rgb = _to_rgb_from_blue_green(blue_f, green_f)
    _plot_rgb_comparison(orig_rgb, filt_rgb, f"{dataset} / {stem}", out_img_dir / "compare_original_filtered_diff.png")

    print("Saved outputs:", out_img_dir)


if __name__ == "__main__":
    main()