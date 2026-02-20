from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import imageio.v3 as iio
import zarr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage.metrics import structural_similarity as ssim
from PFT.core_prog_parts.decoder_omezar import ome_zarr_to_n2v_2d_stack
from PFT.core_prog_parts.n2v_denoising import denoise_2d_for_cellpose, ome_zarr_to_n2v_2d_stack, normalize_yxc_for_n2v
from PFT.core_prog_parts.ome_zarr import save_ome_zarr_next_to_outputs

"""Validation of N2V denoising on 2D datasets."""

TRAIN_DIRNAME = "training_data"
VAL_DIRNAME = "validation_data"
SCALEBAR_UM = 4.0 


def _repo_root_from_this_file() -> Path:
    """
    Root from file location: 

    """
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "scripts":
        return here.parent.parent
    return Path.cwd().resolve()


@dataclass
class SimpleMeta:
    source_path: str = ""
    channel_names: list[str] | None = None
    pixel_size_um_x: float | None = None
    pixel_size_um_y: float | None = None
    pixel_size_um_z: float | None = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dataset_choice() -> str:
    options = ["2d_time", "2d_wga_dapi"]
    print("Choose dataset to validate:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        s = input("Enter 1 or 2: ").strip()
        if s in ("1", "2"):
            return options[int(s) - 1]
        print("Invalid input. Please enter 1 or 2.")


def list_samples(dataset_root: Path, split_dirname: str) -> List[Path]:
    base = dataset_root / split_dirname
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def percentile_norm01(img: np.ndarray, lo: float = 1.0, hi: float = 99.8) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    a = np.percentile(img, lo)
    b = np.percentile(img, hi)
    if b <= a:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - a) / (b - a)
    return np.clip(out, 0.0, 1.0)


def read_pixel_size_um_xy(zarr_dir: Path) -> tuple[float | None, float | None]:
    """
    Try to read pixel size (µm/px) from OME-Zarr multiscales metadata.

    Returns (px_um_y, px_um_x). If not found -> (None, None).
    """
    try:
        root = zarr.open(str(zarr_dir), mode="r")
        ms = root.attrs.get("multiscales", None)
        if not ms:
            return None, None

        ds0 = ms[0].get("datasets", [])[0]
        cts = ds0.get("coordinateTransformations", [])
        if not cts:
            return None, None

        scale = cts[0].get("scale", None)
        if not scale:
            return None, None

        axes = ms[0].get("axes", None)
        if not axes:
            return None, None

        names: list[str] = []
        for a in axes:
            if isinstance(a, dict):
                names.append(a.get("name"))
            else:
                names.append(str(a))

        if "y" in names and "x" in names:
            iy = names.index("y")
            ix = names.index("x")
            px_um_y = float(scale[iy])
            px_um_x = float(scale[ix])
            return px_um_y, px_um_x

        return None, None
    except Exception:
        return None, None


def save_rgb_png_with_scalebar(out_png: Path, den_yxc: np.ndarray, sample_zarr_dir: Path) -> None:
    """
    Save  denoised image.

    - If (Y,X,1): blue
    - If (Y,X,2): ch0=blue(DAPI), ch1=green(WGA)
    """
    if out_png.exists():
        return

    den_yxc = np.asarray(den_yxc)
    if den_yxc.ndim != 3:
        raise ValueError(f"Expected denoised YXC, got {den_yxc.shape}")

    y, x, c = den_yxc.shape
    rgb = np.zeros((y, x, 3), dtype=np.float32)

    if c == 1:
        blue = percentile_norm01(den_yxc[..., 0])
        rgb[..., 2] = blue
    elif c == 2:
        blue = percentile_norm01(den_yxc[..., 0])   # DAPI
        green = percentile_norm01(den_yxc[..., 1])  # WGA
        rgb[..., 2] = blue
        rgb[..., 1] = green
    else:
        gray = percentile_norm01(den_yxc[..., 0])
        rgb[:] = gray[..., None]

    px_um_y, px_um_x = read_pixel_size_um_xy(sample_zarr_dir)
    if px_um_x is None or px_um_x <= 0:
        print(f"[WARN] No pixel size found in {sample_zarr_dir.name}; saving PNG without scalebar.")
        rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
        ensure_dir(out_png.parent)
        iio.imwrite(out_png, rgb_u8)
        return

    scalebar_px = int(round(SCALEBAR_UM / px_um_x))
    if scalebar_px < 1:
        print(f"[WARN] scalebar_px < 1 for {sample_zarr_dir.name}; saving PNG without scalebar.")
        rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
        ensure_dir(out_png.parent)
        iio.imwrite(out_png, rgb_u8)
        return

    ensure_dir(out_png.parent)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_axis_off()

    fontprops = fm.FontProperties(size=10)
    sb = AnchoredSizeBar(
        ax.transData,
        scalebar_px,                 
        f"{SCALEBAR_UM:.0f} µm",      
        loc="lower right",
        pad=0.3,
        borderpad=0.5,
        sep=5,
        frameon=True,
        size_vertical=max(2, scalebar_px // 20),
        fontproperties=fontprops,
    )
    ax.add_artist(sb)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def compute_metrics(raw_yxc: np.ndarray, den_yxc: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute SSIM, MSE, MAE.
    For 2-channel data: compute per-channel and average.
    """
    raw_yxc = np.asarray(raw_yxc, dtype=np.float32)
    den_yxc = np.asarray(den_yxc, dtype=np.float32)

    if raw_yxc.shape != den_yxc.shape:
        raise ValueError(f"raw and den must have same shape. raw={raw_yxc.shape}, den={den_yxc.shape}")

    if raw_yxc.ndim != 3:
        raise ValueError(f"Expected YXC arrays. raw={raw_yxc.shape}, den={den_yxc.shape}")

    c = raw_yxc.shape[-1]
    s_list, mse_list, mae_list = [], [], []

    for ch in range(c):
        r = raw_yxc[..., ch]
        d = den_yxc[..., ch]

        dr = float(r.max() - r.min()) if r.max() > r.min() else 1.0
        s_list.append(float(ssim(r, d, data_range=dr)))
        mse_list.append(float(np.mean((r - d) ** 2)))
        mae_list.append(float(np.mean(np.abs(r - d))))

    return float(np.mean(s_list)), float(np.mean(mse_list)), float(np.mean(mae_list))


def load_raw_yxc(sample_zarr_dir: Path, dataset: str) -> np.ndarray:
    """
    Load raw data in YXC form (single frame).

    - 2d_time: returns (Y,X,1) from channel 0
    - 2d_wga_dapi: returns (Y,X,2) where:
        ch0 = DAPI (blue) = channel 0 in OME-Zarr
        ch1 = WGA  (green)= channel 1 in OME-Zarr
    """
    if dataset == "2d_time":
        raw_stack = ome_zarr_to_n2v_2d_stack(sample_zarr_dir, channel=0, normalize=None)  # (N,Y,X,1)
        raw_yx = raw_stack[0, ..., 0]  # first frame
        return raw_yx[..., None]       # (Y,X,1)

    raw_dapi_stack = ome_zarr_to_n2v_2d_stack(sample_zarr_dir, channel=0, normalize=None)
    raw_wga_stack  = ome_zarr_to_n2v_2d_stack(sample_zarr_dir, channel=1, normalize=None)
    raw_dapi = raw_dapi_stack[0, ..., 0]
    raw_wga  = raw_wga_stack[0, ..., 0]
    return np.stack([raw_dapi, raw_wga], axis=-1)  # (Y,X,2) (DAPI, WGA)


def save_denoised_omezarr(out_dir: Path, den_yxc: np.ndarray, dataset: str, source_path: str) -> None:
    """
    Save denoised as OME-Zarr

      - single channel: save (Y,X)
      - two channels:   save (C,Y,X) with channel_names length=2
    """
    out_zarr = out_dir / "image.ome.zarr"
    if out_zarr.exists():
        return

    den_yxc = np.asarray(den_yxc, dtype=np.float32)

    if dataset == "2d_time":
        arr_to_save = den_yxc[..., 0]  # (Y,X)
        meta = SimpleMeta(source_path=source_path, channel_names=["blue"])
    else:
        # (Y,X,2) -> (C,Y,X)
        arr_to_save = np.transpose(den_yxc, (2, 0, 1))  # (2,Y,X)
        meta = SimpleMeta(source_path=source_path, channel_names=["blue_DAPI", "green_WGA"])

    ensure_dir(out_dir)
    save_ome_zarr_next_to_outputs(out_dir, arr_to_save, meta=meta, overwrite=False)


def main() -> None:
    root = _repo_root_from_this_file()
    training_base = root / "results" / "training_files"
    out_img_base = root / "results" / "N2V_denoised" / "img"
    out_metrics_dir = root / "results" / "N2V_denoised" / "metrics"

    ensure_dir(out_img_base)
    ensure_dir(out_metrics_dir)

    dataset = dataset_choice()
    dataset_root = training_base / dataset

    # Collect all samples
    jobs: List[Tuple[str, str, Path]] = []
    for split_dirname, split_label in [(TRAIN_DIRNAME, "train"), (VAL_DIRNAME, "val")]:
        for sdir in list_samples(dataset_root, split_dirname):
            jobs.append((split_dirname, split_label, sdir))

    if not jobs:
        print(f"No samples found for dataset '{dataset}' under {dataset_root}")
        return

    total = len(jobs)

    metrics_txt = out_metrics_dir / f"{dataset}_metrics.txt"
    mode = "a" if metrics_txt.exists() else "w"

    lines: List[str] = []
    if mode == "w":
        lines.append("sample\tsplit\tSSIM\tMSE\tMAE\n")

    for i, (_split_dirname, split_label, sample_zarr_dir) in enumerate(jobs, 1):
        sample_name = sample_zarr_dir.name
        out_dir = out_img_base / dataset / split_label / sample_name
        out_png = out_dir / "denoised_rgb.png"
        out_zarr = out_dir / "image.ome.zarr"
        out_metrics = out_dir / "metrics.txt"

        # Skip
        if out_png.exists() and out_zarr.exists() and out_metrics.exists():
            print(f"[{i}/{total}] SKIP {split_label} {sample_name}")
            continue

        print(f"[{i}/{total}] {split_label} {sample_name}")

        # Load no normalization
        raw = load_raw_yxc(sample_zarr_dir, dataset=dataset)

        # Denoise 
        den = denoise_2d_for_cellpose(raw)

        # Compute metrics against the normalized input scale the model saw
        raw_norm = normalize_yxc_for_n2v(raw)
        s_val, mse_val, mae_val = compute_metrics(raw_norm, den)

        # Save
        save_rgb_png_with_scalebar(out_png, den, sample_zarr_dir)
        save_denoised_omezarr(out_dir, den, dataset=dataset, source_path=str(sample_zarr_dir))

        ensure_dir(out_dir)
        with open(out_metrics, "w", encoding="utf-8") as f:
            f.write(f"dataset: {dataset}\n")
            f.write(f"split: {split_label}\n")
            f.write(f"sample: {sample_name}\n")
            f.write(f"SSIM: {s_val:.6f}\n")
            f.write(f"MSE:  {mse_val:.8f}\n")
            f.write(f"MAE:  {mae_val:.8f}\n")

        lines.append(f"{sample_name}\t{split_label}\t{s_val:.6f}\t{mse_val:.8f}\t{mae_val:.8f}\n")

    if lines:
        with open(metrics_txt, mode, encoding="utf-8") as f:
            f.writelines(lines)

    print("\nDone.")
    print(f"Metrics: {metrics_txt}")
    print(f"Images:  {out_img_base / dataset}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
