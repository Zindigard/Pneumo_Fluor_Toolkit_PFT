from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from PFT.core_prog_parts.common_paths import filtered_img_root
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.image_utils import (
    normalize01_percentile as norm01_percentile,
    to_uint8_minmax,
    to_uint8_percentile,
)
from PFT.core_prog_parts.notch_filter import list_omezarr_images, _ensure_cyx, _to_numpy
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs

EPS = 1e-12

try:
    from scipy.ndimage import maximum_filter
except Exception:
    maximum_filter = None


def build_base_intensity(img: np.ndarray) -> np.ndarray:
    """Create a normalized intensity image used for thresholding."""
    return norm01_percentile(img)


def classify_base_intensity(
    base_n: np.ndarray,
    t_low: float = 0.33,
    t_high: float = 0.66,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split pixels into low, mid, and high intensity groups."""
    low_mask = base_n < t_low
    mid_mask = (base_n >= t_low) & (base_n < t_high)
    high_mask = base_n >= t_high
    return low_mask, mid_mask, high_mask


def dilate_mask(mask: np.ndarray, size: int = 1) -> np.ndarray:
    """Expand a mask slightly to keep nearby pixels together."""
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


def high_band_weights(
    base_n: np.ndarray,
    high_mask: np.ndarray,
    t_high: float,
) -> np.ndarray:
    """Assign stronger weights to the brightest pixels."""
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
    """Apply thresholding rules to one image plane."""
    name_l = image_name.lower()
    is_120min = "120min" in name_l

    low_mask, mid_mask, high_mask = classify_base_intensity(
        base_n,
        t_low=t_low,
        t_high=t_high,
    )

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


def extract_all_channels_and_axes(zarr_path: Path) -> tuple[np.ndarray, str]:
    """Load one OME-Zarr image and return data with axes."""
    arr, axes = load_ome_zarr(zarr_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    return np.asarray(x, dtype=np.float32), axes


def make_rgb_raw(dataset: str, x: np.ndarray, axes: str) -> np.ndarray:
    """Build an RGB image using raw intensity scaling."""
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


def make_rgb_norm(dataset: str, x: np.ndarray, axes: str) -> np.ndarray:
    """Build an RGB image using display normalization."""
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


def save_thresholded_image(
    dataset: str,
    in_path: Path,
    *,
    kernel_size: int,
    t_low: float,
    t_high: float,
    mid_over_low_factor: float,
    overwrite: bool = False,
) -> Path:
    """Create and save thresholded outputs for one image."""
    stem = in_path.parent.name
    out_dir = filtered_img_root() / dataset / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_zarr = out_dir / "image.ome.zarr"
    raw_tiff = out_dir / f"{stem}_raw.tiff"
    norm_tiff = out_dir / f"{stem}_normalized.tiff"

    if out_zarr.exists() and raw_tiff.exists() and norm_tiff.exists() and not overwrite:
        return out_zarr

    x, axes = extract_all_channels_and_axes(in_path)
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

    tifffile.imwrite(raw_tiff, make_rgb_raw(dataset, y, axes), photometric="rgb")
    tifffile.imwrite(norm_tiff, make_rgb_norm(dataset, y, axes), photometric="rgb")
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
    """Create thresholded cached images for a dataset when needed."""
    src_zarrs = list_omezarr_images(dataset)
    if not src_zarrs:
        raise SystemExit(f"No source OME-Zarr images found for dataset={dataset}")

    prepared: list[Path] = []
    for i, in_path in enumerate(src_zarrs, start=1):
        stem = in_path.parent.name
        out_zarr = save_thresholded_image(
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