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

"Interactive tool to apply notch filters in the frequency domain to OME-Zarr images, with visualization and saving of results."


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
    """
    if dataset == "2d_wga_dapi":
        return []

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
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g))


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
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _to_rgb_from_blue_green(blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    B = _norm01_percentile(blue)
    G = _norm01_percentile(green) if green is not None else np.zeros_like(B)
    R = np.zeros_like(B)
    return np.stack([R, G, B], axis=-1)


def _make_crimson_cmap():
    return viz.make_crimson_cmap()


def _plot_rgb_comparison(orig_rgb: np.ndarray, filt_rgb: np.ndarray, title: str, out_png: Path):
    viz.plot_rgb_comparison(
        orig_rgb,
        filt_rgb,
        title,
        out_png,
        original_label="Original (RGB, display-norm)",
        filtered_label="Filtered (RGB, display-norm)",
    )



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


def angle_energy_profile(
    mean_logmag: np.ndarray,
    *,
    r_min: int,
    r_max: int | None = None,
    n_bins: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
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
    return {"blue": "Blues", "green": "Greens"}.get(name, "gray")


def _imshow_fft(ax, fft_img: np.ndarray, cmap: str):
    viz.imshow_percentile(ax, fft_img, cmap=cmap)


def _pick_radius_centered(mean_fft: np.ndarray, title: str, cmap: str) -> int:
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
        raise RuntimeError("DC radius selection cancelled.")
    return int(selected["r"])


def _save_fft_with_dc_overlay(fft_img: np.ndarray, r_dc: int, out_png: Path, title: str, cmap: str, linewidth: int = 3):
    viz.save_fft_with_dc_overlay(fft_img, r_dc, out_png, title, cmap, linewidth=linewidth)



def _prompt_floats(prompt: str, default: list[float]) -> list[float]:
    s = input(prompt).strip()
    if not s:
        return default
    s = s.replace(",", " ")
    return [float(tok) for tok in s.split()]


def _prompt_float(prompt: str, default: float) -> float:
    s = input(prompt).strip()
    return default if not s else float(s)


def _prompt_int(prompt: str, default: int) -> int:
    s = input(prompt).strip()
    return default if not s else int(s)


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    s = input(prompt).strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true", "t")


def _choose_dataset_interactive() -> str:
    print("Choose dataset:")
    print("  1) 2d_time (blue only)")
    print("  2) 2d_wga_dapi (blue + green)")
    ans = input("Choose number [1/2]: ").strip() or "1"
    return "2d_wga_dapi" if ans == "2" else "2d_time"


def _choose_mode_interactive() -> str:
    print("Choose mode:")
    print("  1) Process ONE image directly (enter parameters first, skip tuning figures)")
    print("  2) Tune on curated subset first, then process curated subset")
    print("  3) Process curated subset directly (enter parameters first, skip tuning figures)")
    ans = input("Choose number [1/2/3]: ").strip() or "2"
    return ans if ans in {"1", "2", "3"} else "2"


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


def _write_single_metrics_report(
    rep: Path,
    dataset: str,
    stem: str,
    params: NotchParams,
    metrics_before: dict,
    metrics_after: dict,
) -> None:
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
    batch_rows: list[dict] = []
    for p in curated_paths:
        idx = _path_to_index(zarrs, p)
        out_dir, before, after = _process_one_image(dataset, idx, zarrs, params, fft_size)
        batch_rows.extend(_rows_for_batch(p.parent.name, before, after))

    dataset_out = results_filters_dir() / "Notch" / dataset
    _write_batch_summary_txt(dataset_out / "batch_metrics_curated_subset.txt", batch_rows, dataset, params)
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