from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import list_omezarr_images, _ensure_cyx, _to_numpy

EPS = 1e-12


INTENSITY_RGB_CMAP = LinearSegmentedColormap.from_list(
    "intensity_rgb_orange_peak",
    [
        (0.00, (0.0, 0.0, 1.0)),   # low -> blue
        (0.55, (0.0, 1.0, 0.0)),   # mid -> green
        (0.85, (1.0, 0.0, 0.0)),   # high -> red
        (1.00, (1.0, 0.65, 0.0)),  # very high -> orange
    ],
)

CURATED_TEST_STEMS: dict[str, list[str]] = {
    "2d_time": [
        "WT_HADA_NHS_5min_ROI1_SIM²",
        "WT_HADA_NHS_5min_ROI2_SIM²",
        "WT_HADA_NHS_5min_ROI3_SIM²",
        "WT_HADA_NHS_5min_ROI4_SIM",
        "WT_HADA_NHS_5min_ROI4_SIM²",
        "WT_HADA_NHS_20min_ROI1_SIM",
        "WT_HADA_NHS_20min_ROI1_SIM²",
        "WT_HADA_NHS_20min_ROI2_SIM",
        "WT_HADA_NHS_20min_ROI2_SIM²",
        "WT_HADA_NHS_20min_ROI3_SIM",
        "WT_HADA_THY_40min_ROI3_SIM²",
        "WT_HADA_THY_40min_ROI4_SIM²",
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


def analysis_root() -> Path:
    out = project_root() / "results" / "analysis" / "intensity_fft"
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


def fft_log_magnitude(img2d: np.ndarray) -> np.ndarray:
    x = np.asarray(img2d, dtype=np.float32)
    F = np.fft.fftshift(np.fft.fft2(x - float(np.mean(x))))
    mag = np.log1p(np.abs(F)).astype(np.float32)
    return norm01_percentile(mag, p_lo=1.0, p_hi=99.8)


def list_curated_test_images(dataset: str, zarrs: list[Path]) -> list[Path]:
    want = CURATED_TEST_STEMS.get(dataset, [])
    by_stem = {p.parent.name: p for p in zarrs}
    return [by_stem[s] for s in want if s in by_stem]


def _extract_display_plane(x: np.ndarray, axes: str, channel_index: int) -> np.ndarray:
    if "c" in axes:
        plane = np.take(x, indices=channel_index, axis=axes.index("c"))
    else:
        plane = x
    if "t" in axes and plane.ndim == 3:
        plane = plane[0]
    return np.asarray(plane, dtype=np.float32)


def load_planes(zarr_path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    arr, axes = load_ome_zarr(zarr_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    blue = _extract_display_plane(x, axes, 0)
    green = _extract_display_plane(x, axes, 1) if dataset == "2d_wga_dapi" and ("c" in axes and x.shape[axes.index("c")] > 1) else None
    return blue, green


def make_rgb(dataset: str, blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    b = to_uint8_percentile(blue)
    z = np.zeros_like(b, dtype=np.uint8)
    if dataset == "2d_wga_dapi" and green is not None:
        g = to_uint8_percentile(green)
        return np.stack([z, g, b], axis=-1)
    return np.stack([z, z, b], axis=-1)


def choose_source_zarr(dataset: str, stem: str, raw_zarr: Path) -> Path:
    filtered_zarr = filtered_img_root() / dataset / stem / "image.ome.zarr"
    return filtered_zarr if filtered_zarr.exists() else raw_zarr


def save_single_channel_outputs(channel: np.ndarray, out_dir: Path, stem: str, channel_name: str) -> None:
    intensity = norm01_percentile(channel)
    fft_img = fft_log_magnitude(channel)

    plt.imsave(out_dir / f"{stem}_{channel_name}_intensity_map.png", intensity, cmap=INTENSITY_RGB_CMAP, vmin=0.0, vmax=1.0)
    plt.imsave(out_dir / f"{stem}_{channel_name}_fft.png", fft_img, cmap="gray", vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")
    axes[0].imshow(intensity, cmap=INTENSITY_RGB_CMAP, vmin=0.0, vmax=1.0)
    axes[0].set_title(f"{channel_name} intensity", color="white")
    axes[1].imshow(fft_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{channel_name} FFT", color="white")
    fig.suptitle(stem, color="white")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_{channel_name}_intensity_fft_panel.png", dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def save_rgb_overview(dataset: str, blue: np.ndarray, green: np.ndarray | None, out_dir: Path, stem: str) -> None:
    rgb = make_rgb(dataset, blue, green)
    plt.imsave(out_dir / f"{stem}_normalized_rgb.png", rgb)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgb)
    ax.set_title(f"{stem} normalized RGB", color="white")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_normalized_rgb_panel.png", dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def process_one(dataset: str, raw_zarr: Path) -> Path:
    stem = raw_zarr.parent.name
    source_zarr = choose_source_zarr(dataset, stem, raw_zarr)
    out_dir = analysis_root() / dataset / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    blue, green = load_planes(source_zarr, dataset)
    save_rgb_overview(dataset, blue, green, out_dir, stem)
    save_single_channel_outputs(blue, out_dir, stem, "blue")
    if dataset == "2d_wga_dapi" and green is not None:
        save_single_channel_outputs(green, out_dir, stem, "green")

    with open(out_dir / "source_used.txt", "w", encoding="utf-8") as f:
        f.write(str(source_zarr) + "\n")

    return out_dir


def choose_dataset_interactive() -> str:
    print("Choose dataset:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    while True:
        s = input("Choose number [1/2]: ").strip()
        if s == "1":
            return "2d_time"
        if s == "2":
            return "2d_wga_dapi"
        print("Please choose 1 or 2.")


def choose_mode_interactive(n: int) -> str:
    print("Choose mode:")
    print("  1) One image")
    print("  2) Curated subset")
    print("  3) All images")
    while True:
        s = input("Choose number [1/2/3]: ").strip()
        if s in {"1", "2", "3"}:
            return s
        print("Please choose 1, 2 or 3.")


def choose_index_interactive(paths: list[Path]) -> int:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Create intensity maps and FFT images from the same images used in filtering.")
    ap.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi"], required=False)
    ap.add_argument("--mode", choices=["one", "curated", "all"], required=False)
    ap.add_argument("--image-index", type=int, default=None)
    args = ap.parse_args()

    dataset = args.dataset or choose_dataset_interactive()
    zarrs = list_omezarr_images(dataset)
    if not zarrs:
        raise FileNotFoundError(f"No OME-Zarr images found for dataset: {dataset}")

    mode = args.mode
    if mode is None:
        mode_choice = choose_mode_interactive(len(zarrs))
        mode = {"1": "one", "2": "curated", "3": "all"}[mode_choice]

    if mode == "curated":
        selected = list_curated_test_images(dataset, zarrs)
        if not selected:
            raise FileNotFoundError(f"No curated images found for dataset: {dataset}")
    elif mode == "all":
        selected = zarrs
    else:
        if args.image_index is None:
            idx = choose_index_interactive(zarrs)
        else:
            idx = args.image_index
        if not (0 <= idx < len(zarrs)):
            raise IndexError(f"image-index must be between 0 and {len(zarrs)-1}")
        selected = [zarrs[idx]]

    print(f"Processing {len(selected)} image(s) from dataset: {dataset}")
    for p in selected:
        out_dir = process_one(dataset, p)
        print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
