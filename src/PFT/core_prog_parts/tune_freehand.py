from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import list_omezarr_images, _ensure_cyx, _to_numpy
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.free_hand_filter import results_filters_dir

try:
    from scipy.ndimage import maximum_filter
except Exception:
    maximum_filter = None

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

def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "src").exists():
            return p
        if (p / "src" / "PFT").exists() and (p / "results").exists():
            return p
    return here.parents[3]


def filtered_img_root() -> Path:
    out = project_root() / "results" / "img" / "filtered"
    out.mkdir(parents=True, exist_ok=True)
    return out


def norm01_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def to_uint8_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    return np.clip(255.0 * norm01_percentile(x, p_lo=p_lo, p_hi=p_hi), 0, 255).astype(np.uint8)


def to_uint8_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip(255.0 * (x - x_min) / (x_max - x_min + EPS), 0, 255).astype(np.uint8)


def build_base_intensity(img: np.ndarray) -> np.ndarray:
    return norm01_percentile(img)


def classify_base_intensity(base_n: np.ndarray, t_low: float = 0.33, t_high: float = 0.66):
    low_mask = base_n < t_low
    mid_mask = (base_n >= t_low) & (base_n < t_high)
    high_mask = base_n >= t_high
    return low_mask, mid_mask, high_mask


def dilate_mask(mask: np.ndarray, size: int = 1) -> np.ndarray:
    if size <= 1:
        return mask.copy()
    if maximum_filter is not None:
        return maximum_filter(mask.astype(np.uint8), size=size) > 0
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(size):
        for dx in range(size):
            y0 = min(dy, h)
            x0 = min(dx, w)
            out[: h - y0, : w - x0] |= mask[y0:, x0:]
    return out


def high_band_weights(base_n: np.ndarray, high_mask: np.ndarray, t_high: float) -> np.ndarray:
    weights = np.zeros_like(base_n, dtype=np.float32)
    if not np.any(high_mask):
        return weights
    rel = (base_n[high_mask] - t_high) / (1.0 - t_high + EPS)
    rel = np.clip(rel, 0.0, 1.0)
    w = np.empty_like(rel, dtype=np.float32)
    w[rel < 0.50] = 0.2
    w[(rel >= 0.50) & (rel < 0.60)] = 0.6
    w[rel >= 0.60] = 1.0
    weights[high_mask] = w
    return weights


def apply_threshold_filter_single(
    raw_img: np.ndarray,
    base_n: np.ndarray,
    image_name: str,
    dataset: str,
    kernel_size: int = 1,
    t_low: float = 0.33,
    t_high: float = 0.66,
    mid_over_low_factor: float = 1.0,
) -> tuple[np.ndarray, dict]:
    name_l = image_name.lower()
    is_120min = "120min" in name_l

    low_mask, mid_mask, high_mask = classify_base_intensity(base_n, t_low=t_low, t_high=t_high)

    n_low = int(np.count_nonzero(low_mask))
    n_mid = int(np.count_nonzero(mid_mask))
    n_high = int(np.count_nonzero(high_mask))

    out = np.zeros_like(raw_img, dtype=np.float32)

    if dataset == "2d_wga_dapi":
        high_w = high_band_weights(base_n, high_mask, t_high=t_high)
        out[high_mask] = raw_img[high_mask] * high_w[high_mask]

        high_keep = dilate_mask(high_mask, size=kernel_size)
        if kernel_size > 1 and np.any(high_keep):
            if maximum_filter is not None:
                propagated = maximum_filter((raw_img * high_w).astype(np.float32), size=kernel_size)
                propagated[~high_keep] = 0.0
            else:
                propagated = np.zeros_like(raw_img, dtype=np.float32)
                weighted_high = raw_img * high_w
                h, w = raw_img.shape
                for dy in range(kernel_size):
                    for dx in range(kernel_size):
                        y1 = h - min(dy, h)
                        x1 = w - min(dx, w)
                        propagated[:y1, :x1] = np.maximum(
                            propagated[:y1, :x1],
                            weighted_high[min(dy, h):, min(dx, w):],
                        )
                propagated[~high_keep] = 0.0
            out[high_keep] = np.maximum(out[high_keep], propagated[high_keep])

        return out, {
            "n_low": n_low,
            "n_mid": n_mid,
            "n_high": n_high,
            "mid_reclassified": False,
            "is_120min": False,
        }

    if is_120min:
        keep_mask = ~low_mask
        out[keep_mask] = raw_img[keep_mask]
        return out, {
            "n_low": n_low,
            "n_mid": n_mid,
            "n_high": n_high,
            "mid_reclassified": False,
            "is_120min": True,
        }

    mid_reclassified = False
    if n_mid > mid_over_low_factor * max(n_low, 1):
        low_mask = low_mask | mid_mask
        mid_mask = np.zeros_like(mid_mask, dtype=bool)
        mid_reclassified = True

    if np.any(mid_mask):
        mid_rel = (base_n[mid_mask] - t_low) / (t_high - t_low + EPS)
        mid_rel = np.clip(mid_rel, 0.0, 1.0)
        mid_weight = 0.2 + 0.6 * mid_rel
        out[mid_mask] = raw_img[mid_mask] * mid_weight

    high_w = high_band_weights(base_n, high_mask, t_high=t_high)
    out[high_mask] = raw_img[high_mask] * high_w[high_mask]

    high_keep = dilate_mask(high_mask, size=kernel_size)
    if kernel_size > 1 and np.any(high_keep):
        if maximum_filter is not None:
            propagated = maximum_filter((raw_img * high_w).astype(np.float32), size=kernel_size)
            propagated[~high_keep] = 0.0
        else:
            propagated = np.zeros_like(raw_img, dtype=np.float32)
            weighted_high = raw_img * high_w
            h, w = raw_img.shape
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    y1 = h - min(dy, h)
                    x1 = w - min(dx, w)
                    propagated[:y1, :x1] = np.maximum(
                        propagated[:y1, :x1],
                        weighted_high[min(dy, h):, min(dx, w):],
                    )
            propagated[~high_keep] = 0.0
        out[high_keep] = np.maximum(out[high_keep], propagated[high_keep])

    return out, {
        "n_low": n_low,
        "n_mid": n_mid,
        "n_high": n_high,
        "mid_reclassified": mid_reclassified,
        "is_120min": False,
    }


def _extract_all_channels_and_axes(zarr_path: Path) -> tuple[np.ndarray, str]:
    arr, axes = load_ome_zarr(zarr_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    return np.asarray(x, dtype=np.float32), axes


def _make_rgb_raw(dataset: str, x: np.ndarray, axes: str) -> np.ndarray:
    if "c" in axes:
        c_i = axes.index("c")
        blue = np.take(x, indices=0, axis=c_i)
        if blue.ndim == 3 and "t" in axes:
            blue = blue[0]
        if dataset == "2d_wga_dapi" and x.shape[c_i] >= 2:
            green = np.take(x, indices=1, axis=c_i)
            if green.ndim == 3 and "t" in axes:
                green = green[0]
        else:
            green = None
    else:
        blue = x[0] if (x.ndim == 3 and "t" in axes) else x
        green = None

    b = to_uint8_minmax(blue)
    z = np.zeros_like(b, dtype=np.uint8)
    if green is None:
        return np.stack([z, z, b], axis=-1)
    g = to_uint8_minmax(green)
    return np.stack([z, g, b], axis=-1)


def _make_rgb_norm(dataset: str, x: np.ndarray, axes: str) -> np.ndarray:
    if "c" in axes:
        c_i = axes.index("c")
        blue = np.take(x, indices=0, axis=c_i)
        if blue.ndim == 3 and "t" in axes:
            blue = blue[0]
        if dataset == "2d_wga_dapi" and x.shape[c_i] >= 2:
            green = np.take(x, indices=1, axis=c_i)
            if green.ndim == 3 and "t" in axes:
                green = green[0]
        else:
            green = None
    else:
        blue = x[0] if (x.ndim == 3 and "t" in axes) else x
        green = None

    b = to_uint8_percentile(blue)
    z = np.zeros_like(b, dtype=np.uint8)
    if green is None:
        return np.stack([z, z, b], axis=-1)
    g = to_uint8_percentile(green)
    return np.stack([z, g, b], axis=-1)


def _save_thresholded_cache_for_image(
    dataset: str,
    in_path: Path,
    *,
    kernel_size: int,
    t_low: float,
    t_high: float,
    mid_over_low_factor: float,
    overwrite: bool = False,
) -> Path:
    stem = in_path.parent.name
    out_dir = filtered_img_root() / dataset / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = out_dir / "image.ome.zarr"
    raw_tiff = out_dir / f"{stem}_raw.tiff"
    norm_tiff = out_dir / f"{stem}_normalized.tiff"

    if out_zarr.exists() and raw_tiff.exists() and norm_tiff.exists() and not overwrite:
        return out_zarr

    x, axes = _extract_all_channels_and_axes(in_path)
    y = x.astype(np.float32, copy=True)

    if "c" in axes:
        c_i = axes.index("c")
        n_channels = y.shape[c_i]
        target_channels = [0] if dataset == "2d_time" else list(range(min(2, n_channels)))
        for c in target_channels:
            plane = np.take(y, indices=c, axis=c_i)
            if plane.ndim == 3 and "t" in axes:
                out_plane = plane.astype(np.float32, copy=True)
                for t in range(plane.shape[0]):
                    base_n = build_base_intensity(plane[t])
                    out_plane[t], _ = apply_threshold_filter_single(
                        raw_img=np.asarray(plane[t], dtype=np.float32),
                        base_n=base_n,
                        image_name=stem,
                        dataset=dataset,
                        kernel_size=kernel_size,
                        t_low=t_low,
                        t_high=t_high,
                        mid_over_low_factor=mid_over_low_factor,
                    )
                sl = [slice(None)] * y.ndim
                sl[c_i] = c
                y[tuple(sl)] = out_plane
            elif plane.ndim == 2:
                base_n = build_base_intensity(plane)
                filtered_plane, _ = apply_threshold_filter_single(
                    raw_img=np.asarray(plane, dtype=np.float32),
                    base_n=base_n,
                    image_name=stem,
                    dataset=dataset,
                    kernel_size=kernel_size,
                    t_low=t_low,
                    t_high=t_high,
                    mid_over_low_factor=mid_over_low_factor,
                )
                sl = [slice(None)] * y.ndim
                sl[c_i] = c
                y[tuple(sl)] = filtered_plane
            else:
                raise ValueError(f"Unexpected plane ndim={plane.ndim} for axes={axes}")
    else:
        if y.ndim == 3 and "t" in axes:
            out_y = y.astype(np.float32, copy=True)
            for t in range(y.shape[0]):
                base_n = build_base_intensity(y[t])
                out_y[t], _ = apply_threshold_filter_single(
                    raw_img=np.asarray(y[t], dtype=np.float32),
                    base_n=base_n,
                    image_name=stem,
                    dataset=dataset,
                    kernel_size=kernel_size,
                    t_low=t_low,
                    t_high=t_high,
                    mid_over_low_factor=mid_over_low_factor,
                )
            y = out_y
        elif y.ndim == 2:
            base_n = build_base_intensity(y)
            y, _ = apply_threshold_filter_single(
                raw_img=np.asarray(y, dtype=np.float32),
                base_n=base_n,
                image_name=stem,
                dataset=dataset,
                kernel_size=kernel_size,
                t_low=t_low,
                t_high=t_high,
                mid_over_low_factor=mid_over_low_factor,
            )
        else:
            raise ValueError(f"Unexpected image shape {y.shape} for axes={axes}")

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

    tifffile.imwrite(raw_tiff, _make_rgb_raw(dataset, y, axes), photometric="rgb")
    tifffile.imwrite(norm_tiff, _make_rgb_norm(dataset, y, axes), photometric="rgb")
    return out_zarr


def prepare_thresholded_dataset(
    dataset: str,
    *,
    kernel_size: int,
    t_low: float,
    t_high: float,
    mid_over_low_factor: float,
    overwrite: bool = False,
) -> list[Path]:
    src_zarrs = list_omezarr_images(dataset)
    if not src_zarrs:
        raise SystemExit(f"No source OME-Zarr images found for dataset={dataset}")

    prepared: list[Path] = []
    for i, in_path in enumerate(src_zarrs, start=1):
        stem = in_path.parent.name
        out_zarr = _save_thresholded_cache_for_image(
            dataset,
            in_path,
            kernel_size=kernel_size,
            t_low=t_low,
            t_high=t_high,
            mid_over_low_factor=mid_over_low_factor,
            overwrite=overwrite,
        )
        prepared.append(out_zarr)
        print(f"[{i:03d}/{len(src_zarrs):03d}] threshold cache ready: {stem}")
    return prepared


@dataclass
class MaskSpec:
    kind: str
    radius: int
    feather: int = 15
    band_halfwidth: int = 8
    depth: float = 0.7
    fft_rects: list[tuple[int, int, int, int]] | None = None

    @property
    def label(self) -> str:
        if self.kind == "soft":
            base = f"soft_r{self.radius}_f{self.feather}"
        elif self.kind == "outer_cross":
            base = f"outer_cross_r{self.radius}_bw{self.band_halfwidth}_f{self.feather}_d{int(round(self.depth * 100))}"
        elif self.kind == "fft_rect_delete":
            base = "fft_rect_delete"
        else:
            base = f"circle_r{self.radius}"
        if self.kind == "fft_rect_delete" and self.fft_rects:
            parts = [f"x{x}_y{y}_w{w}_h{h}" for (x, y, w, h) in self.fft_rects]
            return f"{base}_{'__'.join(parts)}"
        return base

    @property
    def human_title(self) -> str:
        if self.kind == "soft":
            return f"soft mask (r={self.radius}, feather={self.feather})"
        if self.kind == "outer_cross":
            return (
                "outer cross band-stop "
                f"(protect_r={self.radius}, band_half={self.band_halfwidth}, feather={self.feather}, depth={self.depth:.2f})"
            )
        if self.kind == "fft_rect_delete":
            return f"FFT rectangle delete ({len(self.fft_rects or [])} rects)"
        return f"circle mask (r={self.radius})"


def list_curated_test_images(dataset: str, zarrs: list[Path]) -> list[Path]:
    want = CURATED_TEST_STEMS.get(dataset, [])
    by_stem = {p.parent.name: p for p in zarrs}
    return [by_stem[s] for s in want if s in by_stem]


def missing_curated_test_stems(dataset: str, zarrs: list[Path]) -> list[str]:
    all_stems = {p.parent.name for p in zarrs}
    return [s for s in CURATED_TEST_STEMS.get(dataset, []) if s not in all_stems]


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
    out = {"blue": {"neighbor_corr": neighbor_corr(blue), "fft_peak_score": fft_peak_score(blue), "grad_mag_mean": gradient_mag_mean(blue)}}
    if green is not None:
        out["green"] = {"neighbor_corr": neighbor_corr(green), "fft_peak_score": fft_peak_score(green), "grad_mag_mean": gradient_mag_mean(green)}
    return out


def write_metrics_block(f, title: str, metrics: dict) -> None:
    f.write(f"\n=== {title} ===\n")
    for ch_name, m in metrics.items():
        f.write(f"\n[{ch_name}]\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")


def _to_rgb_from_blue_green(blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    B = norm01_percentile(blue)
    G = norm01_percentile(green) if green is not None else np.zeros_like(B)
    R = np.zeros_like(B)
    return np.stack([R, G, B], axis=-1)


def _scale_shared_raw(orig: np.ndarray, filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = float(min(np.min(orig), np.min(filt)))
    hi = float(max(np.max(orig), np.max(filt)))
    if hi <= lo:
        return np.zeros_like(orig, dtype=np.float32), np.zeros_like(filt, dtype=np.float32)
    return (
        np.clip((orig - lo) / (hi - lo), 0.0, 1.0).astype(np.float32),
        np.clip((filt - lo) / (hi - lo), 0.0, 1.0).astype(np.float32),
    )


def _to_rgb_from_blue_green_raw_shared(blue_orig, green_orig, blue_filt, green_filt):
    b1, b2 = _scale_shared_raw(np.asarray(blue_orig, dtype=np.float32), np.asarray(blue_filt, dtype=np.float32))
    if green_orig is not None and green_filt is not None:
        g1, g2 = _scale_shared_raw(np.asarray(green_orig, dtype=np.float32), np.asarray(green_filt, dtype=np.float32))
    else:
        g1 = np.zeros_like(b1)
        g2 = np.zeros_like(b2)
    return np.stack([np.zeros_like(b1), g1, b1], axis=-1), np.stack([np.zeros_like(b2), g2, b2], axis=-1)


def _make_crimson_cmap():
    return LinearSegmentedColormap.from_list("black_to_crimson", [(0.0, (0.0, 0.0, 0.0)), (0.10, (0.08, 0.0, 0.0)), (0.35, (0.30, 0.0, 0.04)), (0.65, (0.62, 0.02, 0.10)), (1.0, (0.86, 0.08, 0.24))])


def _plot_rgb_comparison(orig_rgb, filt_rgb, title: str, out_png: Path, *, variant_label: str):
    diff = np.abs(filt_rgb.astype(np.float32) - orig_rgb.astype(np.float32))
    diff_map = np.mean(diff, axis=-1)
    scale = float(np.percentile(diff_map, 99.7)) + EPS
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


def _save_both_comparisons(blue_before, green_before, blue_after, green_after, title: str, out_dir: Path):
    raw_orig_rgb, raw_filt_rgb = _to_rgb_from_blue_green_raw_shared(blue_before, green_before, blue_after, green_after)
    _plot_rgb_comparison(raw_orig_rgb, raw_filt_rgb, title, out_dir / "compare_original_filtered_diff_raw.png", variant_label="RGB, shared raw scale")
    _plot_rgb_comparison(_to_rgb_from_blue_green(blue_before, green_before), _to_rgb_from_blue_green(blue_after, green_after), title, out_dir / "compare_original_filtered_diff_normalized.png", variant_label="RGB, display-norm")


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


def mean_fft_magnitude(zarr_paths: list[Path], *, channel_index: int, fft_size: int = 512) -> np.ndarray:
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
        acc = m.astype(np.float64) if acc is None else acc + m.astype(np.float64)
        used += 1
    if acc is None or used == 0:
        raise ValueError("Could not compute mean FFT")
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


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    val = input(f"{prompt.strip()} [{suffix}]: ").strip().lower()
    if not val:
        return default
    return val in {"y", "yes", "1", "true"}


def _prompt_int(prompt: str, default: int) -> int:
    s = input(prompt).strip()
    if not s:
        return default
    return int(s)


def _prompt_float(prompt: str, default: float) -> float:
    s = input(prompt).strip()
    if not s:
        return default
    return float(s)


def _choose_dataset_interactive() -> str:
    print("Choose dataset:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    while True:
        c = input("Choose number [1/2]: ").strip() or "1"
        if c == "1":
            return "2d_time"
        if c == "2":
            return "2d_wga_dapi"
        print("Please choose 1 or 2.")


def _choose_mode_interactive() -> str:
    print("Choose mode:")
    print("  1) One image")
    print("  2) Tune on curated subset using mean FFT")
    print("  3) Run curated subset directly")
    while True:
        c = input("Choose number [1/2/3]: ").strip() or "1"
        if c in {"1", "2", "3"}:
            return c
        print("Please choose 1, 2, or 3.")


def _pick_radius_centered(mean_fft: np.ndarray, title: str, cmap: str) -> int:
    h, w = mean_fft.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    _imshow_fft(ax, mean_fft, cmap=cmap)
    ax.set_title(title + "\nMove mouse, click once to set radius")
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


def _pick_multiple_rectangles_on_fft(fft_img: np.ndarray, title: str, cmap: str) -> list[tuple[int, int, int, int]]:
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    _imshow_fft(ax, fft_img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    selected: list[tuple[int, int, int, int]] = []

    def on_select(eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0, x1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        y0, y1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
        rect = (x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1))
        selected.append(rect)
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, edgecolor="orange", linewidth=2))
        fig.canvas.draw_idle()

    rs = RectangleSelector(ax, on_select, useblit=False, button=[1], minspanx=2, minspany=2, interactive=False)
    plt.show()
    rs.set_active(False)
    plt.close(fig)
    if not selected:
        raise RuntimeError("No FFT rectangles selected.")
    return selected


def _collect_dataset_channel_info(dataset: str, zarrs: list[Path]) -> tuple[list[int], list[str]]:
    if dataset == "2d_time":
        return [0], ["blue"]
    return [0, 1], ["blue", "green"]


def _extract_display_plane(x: np.ndarray, axes: str, channel_index: int) -> np.ndarray:
    if "c" in axes:
        plane = np.take(x, indices=channel_index, axis=axes.index("c"))
    else:
        plane = x
    if "t" in axes and plane.ndim == 3:
        plane = plane[0]
    return np.asarray(plane, dtype=np.float32)


def _load_planes(in_path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    blue = _extract_display_plane(x, axes, 0)
    green = _extract_display_plane(x, axes, 1) if dataset == "2d_wga_dapi" and ("c" in axes and x.shape[axes.index("c")] > 1) else None
    return blue, green


def _build_mask(shape: tuple[int, int], spec: MaskSpec) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if spec.kind == "circle":
        return (rr <= spec.radius).astype(np.float32)

    if spec.kind == "soft":
        mask = np.zeros_like(rr, dtype=np.float32)
        mask[rr <= spec.radius] = 1.0
        if spec.feather > 0:
            ring = (rr > spec.radius) & (rr < spec.radius + spec.feather)
            mask[ring] = 1.0 - (rr[ring] - spec.radius) / float(spec.feather)
        return np.clip(mask, 0.0, 1.0)

    if spec.kind == "outer_cross":
        mask = np.ones_like(rr, dtype=np.float32)
        cross = (np.abs(xx - cx) <= spec.band_halfwidth) | (np.abs(yy - cy) <= spec.band_halfwidth)
        outer = rr > spec.radius
        mask[cross & outer] = max(0.0, 1.0 - float(spec.depth))
        if spec.feather > 0:
            ring = (rr > spec.radius) & (rr < spec.radius + spec.feather)
            ring_alpha = (rr[ring] - spec.radius) / float(spec.feather)
            ring_mask = np.ones_like(ring_alpha, dtype=np.float32)
            ring_cross = ((np.abs(xx[ring] - cx) <= spec.band_halfwidth) | (np.abs(yy[ring] - cy) <= spec.band_halfwidth))
            ring_mask[ring_cross] = 1.0 - float(spec.depth) * ring_alpha[ring_cross]
            mask[ring] = np.minimum(mask[ring], ring_mask)
        return np.clip(mask, 0.0, 1.0)

    if spec.kind == "fft_rect_delete":
        mask = np.ones((h, w), dtype=np.float32)
        for rect in spec.fft_rects or []:
            x0, y0, rw, rh = rect
            x1 = min(w, x0 + rw)
            y1 = min(h, y0 + rh)
            mask[y0:y1, x0:x1] = 0.0
            mx0 = max(0, w - x1)
            my0 = max(0, h - y1)
            mx1 = min(w, mx0 + rw)
            my1 = min(h, my0 + rh)
            mask[my0:my1, mx0:mx1] = 0.0
        return mask

    raise ValueError(f"Unknown spec.kind={spec.kind}")


def _apply_frequency_mask_one_plane(img2d: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
    x = np.asarray(img2d, dtype=np.float32)
    mean_val = float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x - mean_val))
    Y = np.fft.ifft2(np.fft.ifftshift(F * mask_keep))
    return np.asarray(np.real(Y) + mean_val, dtype=np.float32)


def _pick_filter_channels(dataset: str, axes: str, x: np.ndarray) -> list[int]:
    if "c" not in axes:
        return [0]
    if dataset == "2d_time":
        return [0]
    return list(range(min(2, x.shape[axes.index("c")])) )


def _write_single_metrics_report(path: Path, dataset: str, stem: str, spec: MaskSpec, before: dict, after: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\nimage: {stem}\nmask: {spec.human_title}\n")
        write_metrics_block(f, "before", before)
        write_metrics_block(f, "after", after)


def _rows_for_batch(stem: str, spec: MaskSpec, before: dict, after: dict) -> list[dict]:
    rows = []
    for ch_name in before:
        row = {"image": stem, "channel": ch_name, "mask": spec.label}
        for k, v in before[ch_name].items():
            row[f"before_{k}"] = v
        for k, v in after[ch_name].items():
            row[f"after_{k}"] = v
        rows.append(row)
    return rows


def _write_batch_summary_txt(path: Path, rows: list[dict], dataset: str, spec: MaskSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\nmask: {spec.human_title}\n\n")
        if not rows:
            f.write("No rows.\n")
            return
        keys = list(rows[0].keys())
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in keys) + "\n")


def _path_to_index(zarrs: list[Path], target: Path) -> int:
    for i, p in enumerate(zarrs):
        if p == target:
            return i
    raise ValueError(f"Path not found: {target}")


def _save_mean_and_example_fft_overlays(out_dir: Path, mean_ffts: Dict[str, np.ndarray], spec: MaskSpec, example_path: Path, dataset: str, fft_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fft_img in mean_ffts.items():
        _save_fft_with_mask_overlay(fft_img, _build_mask(fft_img.shape, spec), out_dir / f"mean_fft_overlay_{name}.png", f"Mean FFT ({name})", _channel_cmap(name), spec, linewidth=3)
    blue_before, green_before = _load_planes(example_path, dataset)
    _save_fft_with_mask_overlay(_fft_logmag(_center_crop(blue_before, fft_size)), _build_mask(_center_crop(blue_before, fft_size).shape, spec), out_dir / "example_fft_overlay_blue.png", "Example FFT (blue)", _channel_cmap("blue"), spec, linewidth=3)
    if green_before is not None:
        _save_fft_with_mask_overlay(_fft_logmag(_center_crop(green_before, fft_size)), _build_mask(_center_crop(green_before, fft_size).shape, spec), out_dir / "example_fft_overlay_green.png", "Example FFT (green)", _channel_cmap("green"), spec, linewidth=3)


def _mirror_rect_on_fft(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, rw, rh = rect
    h, w = shape
    x1 = x0 + rw - 1
    y1 = y0 + rh - 1
    return int(w - 1 - x1), int(h - 1 - y1), int(rw), int(rh)


def _save_fft_with_mask_overlay(fft_img, mask_keep, out_png: Path, title: str, cmap: str, spec: MaskSpec, linewidth: int = 3):
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
        for rect in spec.fft_rects or []:
            x0, y0, rw, rh = rect
            ax.add_patch(Rectangle((x0, y0), rw, rh, fill=False, edgecolor="orange", linewidth=linewidth))
            mx0, my0, mrw, mrh = _mirror_rect_on_fft(rect, (h, w))
            if (mx0, my0, mrw, mrh) != rect:
                ax.add_patch(Rectangle((mx0, my0), mrw, mrh, fill=False, edgecolor="gold", linewidth=max(1, linewidth - 1), linestyle="--"))
    else:
        ax.add_patch(Circle((cx, cy), spec.radius, fill=False, edgecolor="orange", linewidth=linewidth))
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _prompt_mask_spec_interactive(default_radius: int = 72, default_feather: int = 15, allow_fft_rect_delete: bool = True) -> MaskSpec:
    print("Mask type:")
    print("  1) circle")
    print("  2) soft")
    print("  3) outer_cross")
    if allow_fft_rect_delete:
        print("  4) fft_rect_delete")
    choice = input("Choose number [1/2/3/4]: ").strip() or "1"
    if choice == "2":
        return MaskSpec(kind="soft", radius=_prompt_int(f"Radius [{default_radius}]: ", default_radius), feather=_prompt_int(f"Feather [{default_feather}]: ", default_feather))
    if choice == "3":
        return MaskSpec(
            kind="outer_cross",
            radius=_prompt_int(f"Protected radius [{default_radius}]: ", default_radius),
            feather=_prompt_int(f"Feather [{default_feather}]: ", default_feather),
            band_halfwidth=_prompt_int("Band halfwidth [8]: ", 8),
            depth=_prompt_float("Depth [0.7]: ", 0.7),
        )
    if choice == "4" and allow_fft_rect_delete:
        return MaskSpec(kind="fft_rect_delete", radius=0, feather=0, fft_rects=[])
    return MaskSpec(kind="circle", radius=_prompt_int(f"Radius [{default_radius}]: ", default_radius))


def _prompt_mask_spec_for_selected_radius(radius: int, default_feather: int = 15, default_band_halfwidth: int = 8, default_depth: float = 0.7) -> MaskSpec:
    print("Mask type for selected radius:")
    print("  1) circle")
    print("  2) soft")
    print("  3) outer_cross")
    choice = input("Choose number [1/2/3]: ").strip() or "1"
    if choice == "2":
        return MaskSpec(kind="soft", radius=radius, feather=_prompt_int(f"Feather [{default_feather}]: ", default_feather))
    if choice == "3":
        return MaskSpec(kind="outer_cross", radius=radius, feather=_prompt_int(f"Feather [{default_feather}]: ", default_feather), band_halfwidth=_prompt_int(f"Band halfwidth [{default_band_halfwidth}]: ", default_band_halfwidth), depth=_prompt_float(f"Depth [{default_depth}]: ", default_depth))
    return MaskSpec(kind="circle", radius=radius)


def _process_one_image(dataset: str, idx: int, zarrs: list[Path], spec: MaskSpec, fft_size: int) -> tuple[Path, dict, dict]:
    in_path = zarrs[idx]
    stem = in_path.parent.name
    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    y = x.astype(np.float32, copy=True)
    blue_before = _extract_display_plane(x, axes, 0)
    mask_keep = _build_mask(_center_crop(blue_before, fft_size).shape, spec)
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
    out_root = results_filters_dir() / "Free_hand" / dataset / "thresholded_input" / spec.label
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n_c = x.shape[axes.index("c")] if "c" in axes else 1
    meta = SimpleNamespace(pixel_size_um_x=1.0, pixel_size_um_y=1.0, pixel_size_um_z=1.0, channel_names=[f"ch{i}" for i in range(n_c)], source_path=str(in_path), axes=axes)
    save_ome_zarr_next_to_outputs(out_dir, y, meta, overwrite=True, pyramid_3d=False, pyramid_max_layer=0)
    out_zarr = out_dir / "image.ome.zarr"
    blue_before, green_before = _load_planes(in_path, dataset)
    blue_after, green_after = _load_planes(out_zarr, dataset)
    metrics_before = compute_metrics_per_channel(blue_before, green_before)
    metrics_after = compute_metrics_per_channel(blue_after, green_after)
    _write_single_metrics_report(out_dir / "metrics.txt", dataset, stem, spec, metrics_before, metrics_after)
    _save_both_comparisons(blue_before, green_before, blue_after, green_after, f"{dataset} / {stem} / {spec.human_title}", out_dir)
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
    rects = _pick_multiple_rectangles_on_fft(fft_blue, "Chosen image FFT: draw rectangle(s), close window when done", cmap=_channel_cmap("blue"))
    spec = MaskSpec(kind="fft_rect_delete", radius=0, feather=0, fft_rects=rects)
    print(f"Selected: {spec.human_title}")
    if not _prompt_bool("Apply this FFT rectangle delete now?", default=False):
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
    out_root = results_filters_dir() / "Free_hand" / dataset / "thresholded_input" / spec.label
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n_c = x.shape[axes.index("c")] if "c" in axes else 1
    meta = SimpleNamespace(pixel_size_um_x=1.0, pixel_size_um_y=1.0, pixel_size_um_z=1.0, channel_names=[f"ch{i}" for i in range(n_c)], source_path=str(in_path), axes=axes)
    save_ome_zarr_next_to_outputs(out_dir, y, meta, overwrite=True, pyramid_3d=False, pyramid_max_layer=0)
    out_zarr = out_dir / "image.ome.zarr"
    blue_before, green_before = _load_planes(in_path, dataset)
    blue_after, green_after = _load_planes(out_zarr, dataset)
    metrics_before = compute_metrics_per_channel(blue_before, green_before)
    metrics_after = compute_metrics_per_channel(blue_after, green_after)
    _write_single_metrics_report(out_dir / "metrics.txt", dataset, stem, spec, metrics_before, metrics_after)
    _save_both_comparisons(blue_before, green_before, blue_after, green_after, f"{dataset} / {stem} / {spec.human_title}", out_dir)
    _save_fft_with_mask_overlay(fft_blue, mask_keep, out_dir / "chosen_image_fft_overlay_blue.png", f"Chosen image FFT (blue) with {spec.human_title}", _channel_cmap("blue"), spec, linewidth=3)
    if green_before is not None:
        _save_fft_with_mask_overlay(_fft_logmag(green_before), mask_keep, out_dir / "chosen_image_fft_overlay_green.png", f"Chosen image FFT (green) with {spec.human_title}", _channel_cmap("green"), spec, linewidth=3)
    return out_dir, metrics_before, metrics_after


def _process_curated_subset(dataset: str, zarrs: list[Path], curated_paths: list[Path], spec: MaskSpec, fft_size: int) -> Path:
    batch_rows: list[dict] = []
    for p in curated_paths:
        idx = _path_to_index(zarrs, p)
        out_dir, before, after = _process_one_image(dataset, idx, zarrs, spec, fft_size)
        batch_rows.extend(_rows_for_batch(p.parent.name, spec, before, after))
    dataset_out = results_filters_dir() / "Free_hand" / dataset / "thresholded_input" / spec.label
    _write_batch_summary_txt(dataset_out / "batch_metrics_curated_subset.txt", batch_rows, dataset, spec)
    return dataset_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, choices=["2d_time", "2d_wga_dapi", "2d_dpa_wagi"])
    ap.add_argument("--fft_size", type=int, default=512)
    ap.add_argument("--image_index", type=int, default=None)
    ap.add_argument("--overwrite_threshold_cache", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset or _choose_dataset_interactive()
    if dataset == "2d_dpa_wagi":
        dataset = "2d_wga_dapi"

    kernel_size = 1
    t_low = 0.33
    t_high = 0.66
    mid_over_low_factor = 1.0

    print(f"Preparing thresholded OME-Zarr cache for dataset={dataset}")
    zarrs = prepare_thresholded_dataset(
        dataset,
        kernel_size=kernel_size,
        t_low=t_low,
        t_high=t_high,
        mid_over_low_factor=mid_over_low_factor,
        overwrite=args.overwrite_threshold_cache,
    )

    if not zarrs:
        raise SystemExit(f"No thresholded OME-Zarr images available for dataset={dataset}")

    print(f"Found {len(zarrs)} thresholded images in {dataset}:")
    for i, p in enumerate(zarrs[:20]):
        print(f"  [{i:02d}] {p.parent.name}")
    if len(zarrs) > 20:
        print("  ... (showing first 20)")

    mode = _choose_mode_interactive()

    if mode in {"2", "3"}:
        curated_paths = list_curated_test_images(dataset, zarrs)
        missing = missing_curated_test_stems(dataset, zarrs)
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
        if spec.kind == "fft_rect_delete":
            out_dir, _, _ = _process_one_image_fft_rect_delete(dataset, idx, zarrs)
        else:
            print(f"Selected: {spec.human_title}")
            if not _prompt_bool("Apply this mask now?", default=False):
                print("Cancelled.")
                return
            out_dir, _, _ = _process_one_image(dataset, idx, zarrs, spec, args.fft_size)
        print("Saved outputs:", out_dir)
        return

    tuning_dir = results_filters_dir() / "Free_hand" / dataset / "thresholded_input" / "tuning_curated_subset"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    if mode == "2":
        mean_ffts: Dict[str, np.ndarray] = {}
        for ch, name in zip(channels, channel_names):
            mean_ffts[name] = mean_fft_magnitude(curated_paths, channel_index=ch, fft_size=args.fft_size)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            _imshow_fft(ax, mean_ffts[name], cmap=_channel_cmap(name))
            ax.set_title(f"Mean FFT log-magnitude (thresholded curated subset, {dataset}) - {name}")
            ax.axis("off")
        plt.show()
        choice = "blue"
        if len(channel_names) > 1:
            choice = input("Select channel for circle selection [blue/green] (default blue): ").strip().lower() or "blue"
            if choice not in mean_ffts:
                choice = "blue"
        radius = _pick_radius_centered(mean_ffts[choice], f"Protected center radius on mean FFT ({choice})", cmap=_channel_cmap(choice))
        print(f"Selected protected radius: r={radius}px")
        spec = _prompt_mask_spec_for_selected_radius(radius, default_feather=15, default_band_halfwidth=8, default_depth=0.7)
        print(f"Selected: {spec.human_title}")
        _save_mean_and_example_fft_overlays(tuning_dir, mean_ffts, spec, curated_paths[0], dataset, args.fft_size)
        if not _prompt_bool("Apply this tuned mask and run processing now?", default=False):
            print("Tuning finished without processing. Saved tuning overlays:", tuning_dir)
            return
        dataset_out = _process_curated_subset(dataset, zarrs, curated_paths, spec, args.fft_size)
        print("Saved curated subset outputs:", dataset_out)
        return

    spec = _prompt_mask_spec_interactive(default_radius=72, default_feather=15, allow_fft_rect_delete=False)
    print(f"Selected: {spec.human_title}")
    if not _prompt_bool("Run processing on curated subset now?", default=False):
        print("Cancelled.")
        return
    dataset_out = _process_curated_subset(dataset, zarrs, curated_paths, spec, args.fft_size)
    print("Saved curated subset outputs:", dataset_out)


if __name__ == "__main__":
    main()
