
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import list_omezarr_images, _ensure_cyx, _to_numpy
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.free_hand_filter import results_filters_dir
from PFT.core_prog_parts.BM3D import BM3DParams, apply_bm3d_to_image

try:
    from scipy.ndimage import uniform_filter
except Exception:
    uniform_filter = None

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


ALLOWED_120MIN_EXCEPTIONS = {"WT_HADA_NHS_120min_ROI3_SIM"}


def _is_120min_stem(stem: str) -> bool:
    return ("120min" in stem) and (stem not in ALLOWED_120MIN_EXCEPTIONS)


def norm01_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def to_uint8_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    return np.clip(255.0 * norm01_percentile(x, p_lo=p_lo, p_hi=p_hi), 0, 255).astype(np.uint8)


def _extract_display_plane(x: np.ndarray, axes: str, channel_index: int) -> np.ndarray:
    if "c" in axes:
        plane = np.take(x, indices=channel_index, axis=axes.index("c"))
    else:
        plane = x
    if "t" in axes and plane.ndim == 3:
        plane = plane[0]
    return np.asarray(plane, dtype=np.float32)


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
    from matplotlib.colors import LinearSegmentedColormap
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
    ax2.set_title(f"Thresholded ({variant_label})")
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
    raw_orig_rgb, raw_filt_rgb = _to_rgb_from_blue_green_raw_shared(
        blue_before, green_before, blue_after, green_after
    )
    _plot_rgb_comparison(
        raw_orig_rgb,
        raw_filt_rgb,
        title,
        out_dir / "compare_original_thresholded_diff_raw.png",
        variant_label="RGB, shared raw scale",
    )
    _plot_rgb_comparison(
        _to_rgb_from_blue_green(blue_before, green_before),
        _to_rgb_from_blue_green(blue_after, green_after),
        title,
        out_dir / "compare_original_thresholded_diff_normalized.png",
        variant_label="RGB, display-norm",
    )


def _mean_filter(mask: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(mask, dtype=np.float32)
    if uniform_filter is not None:
        return uniform_filter(x, size=size, mode="constant", cval=0.0)
    pad = size // 2
    xp = np.pad(x, pad, mode="constant", constant_values=0.0)
    out = np.zeros_like(x, dtype=np.float32)
    H, W = x.shape
    area = float(size * size)
    for y in range(H):
        for x0 in range(W):
            out[y, x0] = xp[y:y+size, x0:x0+size].sum() / area
    return out


@dataclass
class LocalHighThresholdSpec:
    high_percentile: float = 99.2
    frac3_keep: float = 0.40
    support3_frac: float = 0.70
    kernel3: int = 3
    kernel10: int = 10
    frac10_keep: float = 0.40

    @property
    def label(self) -> str:
        return (
            f"p{self.high_percentile:.1f}_k3f{int(round(self.frac3_keep*100))}"
            f"_s3{int(round(self.support3_frac*100))}_k10f{int(round(self.frac10_keep*100))}"
        )

    @property
    def human_title(self) -> str:
        return (
            f"local-high threshold "
            f"(high>p{self.high_percentile:.1f}, 3x3>={self.frac3_keep:.2f}, "
            f"support3>={self.support3_frac:.2f}, 10x10>={self.frac10_keep:.2f})"
        )


def apply_local_high_threshold_single(raw_img: np.ndarray, spec: LocalHighThresholdSpec):
    x = np.asarray(raw_img, dtype=np.float32)
    thr = float(np.percentile(x, spec.high_percentile))
    high_mask = x >= thr
    frac3_high = _mean_filter(high_mask, size=spec.kernel3)
    keep3 = high_mask & (frac3_high >= spec.frac3_keep)
    frac3_keep = _mean_filter(keep3, size=spec.kernel3)
    strong_cluster = keep3 & (frac3_keep >= spec.support3_frac)
    frac10_high = _mean_filter(high_mask, size=spec.kernel10)
    keep10 = strong_cluster & (frac10_high >= spec.frac10_keep)
    final_keep = keep3 | keep10

    out = np.zeros_like(x, dtype=np.float32)
    out[final_keep] = x[final_keep]
    info = {
        "high_threshold_raw": thr,
        "n_high": int(np.count_nonzero(high_mask)),
        "n_keep3": int(np.count_nonzero(keep3)),
        "n_strong_cluster": int(np.count_nonzero(strong_cluster)),
        "n_keep10": int(np.count_nonzero(keep10)),
        "n_final": int(np.count_nonzero(final_keep)),
        "frac_final": float(np.mean(final_keep)),
    }
    return out, final_keep, info


def compute_basic_stats(img: np.ndarray) -> dict:
    x = np.asarray(img, dtype=np.float32)
    nz = x[x > 0]
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p99": float(np.percentile(x, 99.0)),
        "nonzero_fraction": float(np.mean(x > 0)),
        "nonzero_mean": float(np.mean(nz)) if nz.size else 0.0,
    }


def _load_planes(zarr_path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    arr, axes = load_ome_zarr(zarr_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    blue = _extract_display_plane(x, axes, 0)
    green = _extract_display_plane(x, axes, 1) if dataset == "2d_wga_dapi" and ("c" in axes and x.shape[axes.index("c")] > 1) else None
    return blue, green


def list_source_images(dataset: str) -> list[Path]:
    zarrs = list_omezarr_images(dataset)
    return [p for p in zarrs if not _is_120min_stem(p.parent.name)]


def list_curated_test_images(dataset: str, zarrs: list[Path]) -> list[Path]:
    want = [s for s in CURATED_TEST_STEMS.get(dataset, []) if not _is_120min_stem(s)]
    by_stem = {p.parent.name: p for p in zarrs}
    return [by_stem[s] for s in want if s in by_stem]


def missing_curated_test_stems(dataset: str, zarrs: list[Path]) -> list[str]:
    want = [s for s in CURATED_TEST_STEMS.get(dataset, []) if not _is_120min_stem(s)]
    all_stems = {p.parent.name for p in zarrs}
    return [s for s in want if s not in all_stems]


def _path_to_index(zarrs: list[Path], target: Path) -> int:
    for i, p in enumerate(zarrs):
        if p == target:
            return i
    raise ValueError(f"Path not found: {target}")


def _pick_filter_channels(dataset: str, axes: str, x: np.ndarray) -> list[int]:
    if "c" not in axes:
        return [0]
    if dataset == "2d_time":
        return [0]
    return list(range(min(2, x.shape[axes.index("c")])))


def _write_metrics(path: Path, dataset: str, stem: str, spec: LocalHighThresholdSpec, before: dict, after: dict, extra: dict, bm3d_applied: bool):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\n")
        f.write(f"image: {stem}\n")
        f.write(f"bm3d_after_threshold: {bm3d_applied}\n")
        f.write(f"filter: {spec.human_title}\n\n")
        f.write("=== threshold info ===\n")
        for k, v in extra.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== before ===\n")
        for ch, vals in before.items():
            f.write(f"\n[{ch}]\n")
            for k, v in vals.items():
                f.write(f"{k}: {v}\n")
        f.write("\n=== after ===\n")
        for ch, vals in after.items():
            f.write(f"\n[{ch}]\n")
            for k, v in vals.items():
                f.write(f"{k}: {v}\n")


def _save_mask_preview(mask: np.ndarray, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, color="white")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _save_single_channel_panel(before: np.ndarray, after: np.ndarray, mask: np.ndarray, out_png: Path, stem: str, channel_name: str):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    for ax, img, ttl in [
        (ax1, before, "Original"),
        (ax2, after, "Thresholded"),
        (ax3, mask.astype(np.float32), "Kept mask"),
    ]:
        vmin = float(np.percentile(img, 1.0))
        vmax = float(np.percentile(img, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        if ttl == "Kept mask":
            vmin, vmax = 0.0, 1.0
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(ttl)
        ax.axis("off")
    fig.suptitle(f"{stem} / {channel_name}")
    fig.patch.set_facecolor("black")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _choose_bm3d_params_interactive() -> BM3DParams:
    def _pf(prompt: str, default: float) -> float:
        s = input(prompt).strip()
        return default if not s else float(s)

    print("BM3D parameters after threshold:")
    sigma = _pf("  sigma_psd [0.15]: ", 0.15)
    post = _pf("  post_gaussian_sigma [0.3]: ", 0.3)
    blend = _pf("  blend_original [0.7]: ", 0.7)
    return BM3DParams(
        sigma_psd=sigma,
        stage="all",
        profile=None,
        post_gaussian_sigma=post,
        blend_original=blend,
        normalize_mode="dtype",
        preserve_range=True,
    )


def _process_one_image(dataset: str, idx: int, zarrs: list[Path], spec: LocalHighThresholdSpec, *, apply_bm3d_after: bool, bm3d_params: BM3DParams | None) -> tuple[Path, dict, dict]:
    in_path = zarrs[idx]
    stem = in_path.parent.name
    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    y = x.astype(np.float32, copy=True)

    channel_names = {0: "blue", 1: "green"}
    before_metrics = {}
    extra_info = {}

    chs = _pick_filter_channels(dataset, axes, x)
    for c in chs:
        sl = [slice(None)] * y.ndim
        if "c" in axes:
            sl[axes.index("c")] = c
        plane = y[tuple(sl)]
        ch_name = channel_names.get(c, f"ch{c}")

        if plane.ndim == 2:
            before_metrics[ch_name] = compute_basic_stats(plane)
            filtered, _, info = apply_local_high_threshold_single(plane, spec)
            y[tuple(sl)] = filtered
            extra_info[ch_name] = info
        elif plane.ndim == 3 and "t" in axes:
            out_plane = plane.astype(np.float32, copy=True)
            first_info = None
            for t in range(plane.shape[0]):
                filtered, _, info = apply_local_high_threshold_single(plane[t], spec)
                out_plane[t] = filtered
                if t == 0:
                    first_info = info
            y[tuple(sl)] = out_plane
            before_metrics[ch_name] = compute_basic_stats(plane[0])
            extra_info[ch_name] = first_info
        else:
            raise ValueError(f"Unexpected plane ndim={plane.ndim} for axes={axes}")

    if apply_bm3d_after and bm3d_params is not None:
        channel_mode = "blue" if dataset == "2d_time" else "auto"
        y = apply_bm3d_to_image(y, axes, bm3d_params, channel_mode=channel_mode)

    after_metrics = {}
    for c in chs:
        sl = [slice(None)] * y.ndim
        if "c" in axes:
            sl[axes.index("c")] = c
        plane = y[tuple(sl)]
        ch_name = channel_names.get(c, f"ch{c}")
        after_metrics[ch_name] = compute_basic_stats(plane if plane.ndim == 2 else plane[0])

    out_root = results_filters_dir() / "Local_high_threshold" / dataset / ("with_BM3D" if apply_bm3d_after else "no_BM3D") / spec.label
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_c = x.shape[axes.index("c")] if "c" in axes else 1
    meta = SimpleNamespace(
        pixel_size_um_x=1.0,
        pixel_size_um_y=1.0,
        pixel_size_um_z=1.0,
        channel_names=[f"ch{i}" for i in range(n_c)],
        source_path=str(in_path),
        axes=axes,
    )
    save_ome_zarr_next_to_outputs(out_dir, y, meta, overwrite=True, pyramid_3d=False, pyramid_max_layer=0)
    tifffile.imwrite(out_dir / f"{stem}_normalized.tiff", _make_rgb_norm(dataset, y, axes), photometric="rgb")

    blue_before, green_before = _load_planes(in_path, dataset)
    blue_after, green_after = _load_planes(out_dir / "image.ome.zarr", dataset)
    _save_both_comparisons(
        blue_before, green_before, blue_after, green_after,
        f"{dataset} / {stem} / {spec.human_title} / BM3D={apply_bm3d_after}",
        out_dir,
    )

    for c in chs:
        ch_name = channel_names.get(c, f"ch{c}")
        before = blue_before if c == 0 else green_before
        after = blue_after if c == 0 else green_after
        _, keep_mask, _ = apply_local_high_threshold_single(before, spec)
        _save_mask_preview(keep_mask.astype(np.float32), out_dir / f"mask_keep_{ch_name}.png", f"{stem} / {ch_name} kept mask")
        _save_single_channel_panel(before, after, keep_mask, out_dir / f"{ch_name}_original_thresholded_mask.png", stem, ch_name)

    merged_extra = {}
    for ch_name, vals in extra_info.items():
        for k, v in vals.items():
            merged_extra[f"{ch_name}_{k}"] = v
    _write_metrics(out_dir / "metrics.txt", dataset, stem, spec, before_metrics, after_metrics, merged_extra, apply_bm3d_after)
    return out_dir, before_metrics, after_metrics


def _write_batch_summary_txt(path: Path, rows: list[dict], dataset: str, spec: LocalHighThresholdSpec, *, apply_bm3d_after: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\nfilter: {spec.human_title}\nbm3d_after_threshold: {apply_bm3d_after}\nskip_120min: True\n\n")
        if not rows:
            f.write("No rows.\n")
            return
        keys = list(rows[0].keys())
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in keys) + "\n")


def _process_curated_subset(dataset: str, zarrs: list[Path], curated_paths: list[Path], spec: LocalHighThresholdSpec, *, apply_bm3d_after: bool, bm3d_params: BM3DParams | None) -> Path:
    rows: list[dict] = []
    for p in curated_paths:
        idx = _path_to_index(zarrs, p)
        _, before, after = _process_one_image(dataset, idx, zarrs, spec, apply_bm3d_after=apply_bm3d_after, bm3d_params=bm3d_params)
        for ch_name in before:
            row = {"image": p.parent.name, "channel": ch_name, "filter": spec.label, "bm3d_after_threshold": apply_bm3d_after}
            for k, v in before[ch_name].items():
                row[f"before_{k}"] = v
            for k, v in after[ch_name].items():
                row[f"after_{k}"] = v
            rows.append(row)
    dataset_out = results_filters_dir() / "Local_high_threshold" / dataset / ("with_BM3D" if apply_bm3d_after else "no_BM3D") / spec.label
    _write_batch_summary_txt(dataset_out / "batch_metrics_curated_subset.txt", rows, dataset, spec, apply_bm3d_after=apply_bm3d_after)
    return dataset_out


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
    print("  2) Curated subset")
    print("  3) All images")
    while True:
        c = input("Choose number [1/2/3]: ").strip() or "1"
        if c in {"1", "2", "3"}:
            return c
        print("Please choose 1, 2 or 3.")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    s = input(f"{prompt} {suffix} ").strip().lower()
    if not s:
        return default
    return s in {"y", "yes", "1", "true"}


def _prompt_float(prompt: str, default: float) -> float:
    s = input(prompt).strip()
    if not s:
        return default
    return float(s)


def _choose_index_interactive(paths: list[Path]) -> int:
    for i, p in enumerate(paths):
        print(f"  [{i:02d}] {p.parent.name}")
    while True:
        s = input(f"Choose image index [0-{len(paths)-1}]: ").strip()
        try:
            idx = int(s)
        except ValueError:
            idx = -1
        if 0 <= idx < len(paths):
            return idx
        print("Invalid index.")


def main():
    ap = argparse.ArgumentParser(description="Local high-intensity thresholding, always skipping 120min images, with optional BM3D after threshold.")
    ap.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi"], required=False)
    ap.add_argument("--mode", choices=["one", "curated", "all"], required=False)
    ap.add_argument("--image-index", type=int, default=None)
    ap.add_argument("--apply-bm3d-after", action="store_true")
    ap.add_argument("--high-percentile", type=float, default=None)
    ap.add_argument("--frac3", type=float, default=None)
    ap.add_argument("--support3", type=float, default=None)
    ap.add_argument("--frac10", type=float, default=None)
    args = ap.parse_args()

    dataset = args.dataset or _choose_dataset_interactive()
    zarrs = list_source_images(dataset)
    if not zarrs:
        raise FileNotFoundError(f"No non-120min OME-Zarr images found for dataset={dataset}")

    print(f"Found {len(zarrs)} images in {dataset} (120min skipped):")
    for i, p in enumerate(zarrs[:20]):
        print(f"  [{i:02d}] {p.parent.name}")
    if len(zarrs) > 20:
        print("  ... (showing first 20)")

    mode = args.mode
    if mode is None:
        mode_choice = _choose_mode_interactive()
        mode = {"1": "one", "2": "curated", "3": "all"}[mode_choice]

    spec = LocalHighThresholdSpec(
        high_percentile=args.high_percentile if args.high_percentile is not None else _prompt_float("High-intensity percentile [99.2]: ", 99.2),
        frac3_keep=args.frac3 if args.frac3 is not None else _prompt_float("3x3 high-pixel fraction to keep [0.40]: ", 0.40),
        support3_frac=args.support3 if args.support3 is not None else _prompt_float("3x3 kept-pixel support for 10x10 check [0.70]: ", 0.70),
        frac10_keep=args.frac10 if args.frac10 is not None else _prompt_float("10x10 high-pixel fraction to keep [0.40]: ", 0.40),
        kernel3=3,
        kernel10=10,
    )
    print(f"Selected: {spec.human_title}")

    apply_bm3d_after = args.apply_bm3d_after or _prompt_yes_no("Apply BM3D after threshold?", default=False)
    bm3d_params = _choose_bm3d_params_interactive() if apply_bm3d_after else None

    if mode == "curated":
        selected = list_curated_test_images(dataset, zarrs)
        missing = missing_curated_test_stems(dataset, zarrs)
        if dataset == "2d_time":
            print(f"Curated subset size: {len(selected)}")
            for p in selected:
                print(f"  - {p.parent.name}")
            if missing:
                print("Missing curated stems:")
                for s in missing:
                    print(f"  - {s}")
        else:
            print("No explicit curated list for 2d_wga_dapi, using all images.")
            selected = zarrs
    elif mode == "all":
        selected = zarrs
    else:
        idx = args.image_index if args.image_index is not None else _choose_index_interactive(zarrs)
        if not (0 <= idx < len(zarrs)):
            raise IndexError(f"image-index must be between 0 and {len(zarrs)-1}")
        selected = [zarrs[idx]]

    print(f"Processing {len(selected)} image(s)...")
    if mode == "one":
        idx = _path_to_index(zarrs, selected[0])
        out_dir, _, _ = _process_one_image(dataset, idx, zarrs, spec, apply_bm3d_after=apply_bm3d_after, bm3d_params=bm3d_params)
        print(f"Saved: {out_dir}")
    elif mode == "curated" and dataset == "2d_time":
        dataset_out = _process_curated_subset(dataset, zarrs, selected, spec, apply_bm3d_after=apply_bm3d_after, bm3d_params=bm3d_params)
        print(f"Saved curated subset outputs: {dataset_out}")
    else:
        rows = []
        for p in selected:
            idx = _path_to_index(zarrs, p)
            out_dir, before, after = _process_one_image(dataset, idx, zarrs, spec, apply_bm3d_after=apply_bm3d_after, bm3d_params=bm3d_params)
            for ch_name in before:
                row = {"image": p.parent.name, "channel": ch_name, "filter": spec.label, "bm3d_after_threshold": apply_bm3d_after}
                for k, v in before[ch_name].items():
                    row[f"before_{k}"] = v
                for k, v in after[ch_name].items():
                    row[f"after_{k}"] = v
                rows.append(row)
            print(f"Saved: {out_dir}")
        dataset_out = results_filters_dir() / "Local_high_threshold" / dataset / ("with_BM3D" if apply_bm3d_after else "no_BM3D") / spec.label
        _write_batch_summary_txt(dataset_out / "batch_metrics.txt", rows, dataset, spec, apply_bm3d_after=apply_bm3d_after)
        print(f"Saved batch summary: {dataset_out / 'batch_metrics.txt'}")


if __name__ == "__main__":
    main()
