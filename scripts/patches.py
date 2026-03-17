from __future__ import annotations

from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
for _p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (_p / "src" / "PFT").exists():
        _SRC_DIR = _p / "src"
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        break

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PFT.core_prog_parts import visualization as viz
from PFT.core_prog_parts.common_paths import ensure_dir
from PFT.core_prog_parts.image_utils import normalize01_percentile, rgb_wga_dapi_norm, rgb_wga_dapi_raw_shared
from matplotlib.widgets import RectangleSelector
import tifffile as tiff

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\training_files\2d_wga_dapi\training_data")
OUTDIR = ensure_dir(ROOT / "normvsnormalized_image")

DEFAULT_LEVEL = 0


def find_omezarr_dirs(root: Path) -> list[Path]:
    zdirs = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / ".zgroup").exists() or (p / "zarr.json").exists():
            zdirs.append(p)


    zdirs = sorted(zdirs, key=lambda x: len(x.parts))
    keep: list[Path] = []
    for d in zdirs:
        if not any(str(d).startswith(str(k) + os.sep) for k in keep):
            keep.append(d)
    return keep


def ensure_cyx_keep_dtype(arr: np.ndarray, axes: str) -> np.ndarray:
    """
    Convert loaded array to shape (C, Y, X) for 2D images.
    If T/Z present -> take index 0.
    If no channel axis -> create C=1.
    DOES NOT change dtype (keeps uint16 if input is uint16).
    """
    axes = axes.lower()
    a = np.asarray(arr)
    wanted = ["c", "y", "x"]

    if "c" not in axes:
        slicer = []
        for ax in axes:
            if ax in ["y", "x"]:
                slicer.append(slice(None))
            else:
                slicer.append(0)
        a2 = a[tuple(slicer)]
        if a2.ndim != 2:
            raise ValueError(f"Expected 2D after slicing, got shape={a2.shape}, axes={axes}")
        return a2[None, :, :]  # (1,Y,X), same dtype

    slicer = []
    for ax in axes:
        if ax in wanted:
            slicer.append(slice(None))
        else:
            slicer.append(0)
    a2 = a[tuple(slicer)]

    axes2 = "".join([ax for ax in axes if ax in wanted])
    perm = [axes2.index("c"), axes2.index("y"), axes2.index("x")]
    a3 = np.transpose(a2, perm)

    if a3.ndim != 3:
        raise ValueError(f"Expected (C,Y,X), got shape={a3.shape}, axes={axes}")
    return a3  # same dtype


def normalize_percentile_to_float01(img2d: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Compatibility wrapper around the shared percentile normalization helper."""
    return normalize01_percentile(img2d, p_lo=p_low, p_hi=p_high)


def normalized_uint16_from_raw(img2d_uint16: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Percentile normalize and map to uint16 [0..65535] for saving.
    """
    x01 = normalize_percentile_to_float01(img2d_uint16, p_low, p_high)  # float32 [0,1]
    return (x01 * 65535.0 + 0.5).astype(np.uint16)


def rgb_raw_uint8_c0blue_c1green(cyx: np.ndarray, vmax: float) -> np.ndarray:
    """Compatibility wrapper using the shared raw RGB composition helper."""
    if cyx.ndim != 3:
        raise ValueError(f"Expected (C,Y,X), got {cyx.shape}")
    C = cyx.shape[0]
    if C >= 2:
        rgb = rgb_wga_dapi_raw_shared(cyx[0], cyx[1])
        return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if C == 1:
        x = np.clip(cyx[0].astype(np.float32) / max(float(vmax), 1e-12), 0.0, 1.0)
        u = (x * 255.0 + 0.5).astype(np.uint8)
        return np.dstack([u, u, u])
    return np.zeros((cyx.shape[1], cyx.shape[2], 3), dtype=np.uint8)


def rgb_norm_float_c0blue_c1green(cyx: np.ndarray) -> np.ndarray:
    """Compatibility wrapper using the shared normalized RGB composition helper."""
    if cyx.ndim != 3:
        raise ValueError(f"Expected (C,Y,X), got {cyx.shape}")
    C = cyx.shape[0]
    if C >= 2:
        return rgb_wga_dapi_norm(cyx[0], cyx[1])
    if C == 1:
        x = normalize01_percentile(cyx[0], p_lo=1, p_hi=99)
        return np.dstack([x, x, x]).astype(np.float32)
    return np.zeros((cyx.shape[1], cyx.shape[2], 3), dtype=np.float32)

def extract_patches_grid(img2d: np.ndarray, patch: int, stride: int, max_patches: int) -> np.ndarray:
    H, W = img2d.shape
    patches = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            patches.append(img2d[y:y + patch, x:x + patch])
            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break
    if not patches:
        return np.zeros((0, patch, patch), dtype=img2d.dtype)
    return np.stack(patches, axis=0)


class Cropper:
    def __init__(self, img: np.ndarray, title: str):
        self.img = img
        self.roi = None  # (rmin,rmax,cmin,cmax)

        self.fig, self.ax = plt.subplots()
        if img.ndim == 3:
            self.ax.imshow(img)  # RGB
        else:
            self.ax.imshow(img, cmap="gray")
        self.ax.set_title(title)
        self.ax.axis("off")

        self.selector = RectangleSelector(
            self.ax,
            onselect=self._on_select,
            useblit=True,
            button=[1],
            interactive=True,
            drag_from_anywhere=True,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        print("\n=== CROPPING CONTROLS ===")
        print("- Drag mouse to draw rectangle")
        print("- ENTER to confirm and save crop")
        print("- ESC to cancel/quit without saving\n")

    def _on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        cmin = int(np.floor(min(x1, x2)))
        cmax = int(np.ceil(max(x1, x2)))
        rmin = int(np.floor(min(y1, y2)))
        rmax = int(np.ceil(max(y1, y2)))
        self.roi = (rmin, rmax, cmin, cmax)

    def _on_key(self, event):
        if event.key == "enter":
            plt.close(self.fig)
        elif event.key == "escape":
            self.roi = None
            plt.close(self.fig)

    def run(self):
        plt.show()
        return self.roi


def ask_int(prompt: str, lo: int, hi: int, default: int | None = None) -> int:
    while True:
        s = input(f"{prompt} [{lo}-{hi}]{' (default '+str(default)+')' if default is not None else ''}: ").strip()
        if s == "" and default is not None:
            return default
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print("Invalid input. Try again.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    s = input(f"{prompt} ({d}): ").strip().lower()
    if s == "":
        return default
    return s in ("y", "yes", "1", "true")


def print_stats_cyx(name: str, cyx: np.ndarray):
    C = cyx.shape[0]
    gmin = int(np.min(cyx))
    gmax = int(np.max(cyx))
    print(f"\n[{name}]")
    print(f"  shape: {tuple(cyx.shape)}")
    print(f"  dtype: {cyx.dtype}")
    print(f"  global min/max: {gmin} / {gmax}")
    for ch in range(C):
        cmin = int(np.min(cyx[ch]))
        cmax = int(np.max(cyx[ch]))
        print(f"  ch{ch} min/max: {cmin} / {cmax}")

def save_comparison_figure(raw_rgb_uint8: np.ndarray, norm_rgb_float: np.ndarray, out_path: Path, title: str):
    viz.save_comparison_figure_tiff(raw_rgb_uint8, norm_rgb_float, out_path, title)

def main():
    print(f"Searching for OME-Zarr datasets under:\n  {ROOT}\n")
    datasets = find_omezarr_dirs(ROOT)
    if not datasets:
        raise RuntimeError(f"No OME-Zarr dirs found under {ROOT}")

    print("Found datasets:")
    for i, p in enumerate(datasets):
        print(f"  [{i}] {p}")

    idx = ask_int("Choose dataset index", 0, len(datasets) - 1)
    zpath = datasets[idx]

    level = ask_int("Choose multiscale level", 0, 10, default=DEFAULT_LEVEL)

    print(f"\nLoading:\n  {zpath}\nLevel={level}\n")
    arr, axes = load_ome_zarr(zpath, level=level, as_numpy=True)

    # Original dtype + axes
    arr_np = np.asarray(arr)
    print(f"Loaded array dtype (as returned by load_ome_zarr): {arr_np.dtype}")
    print(f"Loaded axes: '{axes}'")

    cyx = ensure_cyx_keep_dtype(arr_np, axes)  
    print_stats_cyx("FULL IMAGE (CYX, original dtype)", cyx)

    C, H, W = cyx.shape
    if C >= 2:
        print("\nComposite mapping for display: Channel 0 = BLUE, Channel 1 = GREEN")

    default_vmax = 65535 if cyx.dtype == np.uint16 else int(max(1, int(np.max(cyx))))
    vmax = ask_int("RAW display vmax (linear). Use 65535 for uint16", 1, 10_000_000, default=default_vmax)
    vmax_f = float(vmax)

    raw_rgb_uint8 = rgb_raw_uint8_c0blue_c1green(cyx, vmax=vmax_f)

    cropper = Cropper(
        raw_rgb_uint8,
        title=f"{zpath.name} | level {level} | RAW RGB (C0=Blue, C1=Green) | linear vmax={vmax}"
    )
    roi = cropper.run()
    if roi is None:
        print("Cancelled. No files saved.")
        return

    rmin, rmax, cmin, cmax = roi
    rmin = max(0, min(H - 1, rmin))
    rmax = max(rmin + 1, min(H, rmax))
    cmin = max(0, min(W - 1, cmin))
    cmax = max(cmin + 1, min(W, cmax))

    raw_crop_cyx = cyx[:, rmin:rmax, cmin:cmax] 
    print_stats_cyx("CROP (RAW_CYX, original dtype)", raw_crop_cyx)
    print(f"\nCrop ROI: rows [{rmin}:{rmax}], cols [{cmin}:{cmax}]")

    norm_crop_cyx = np.stack(
        [normalized_uint16_from_raw(raw_crop_cyx[i], 1, 99) for i in range(raw_crop_cyx.shape[0])],
        axis=0
    ).astype(np.uint16)

    raw_crop_rgb_uint8 = rgb_raw_uint8_c0blue_c1green(raw_crop_cyx, vmax=vmax_f)
    norm_crop_rgb_float = rgb_norm_float_c0blue_c1green(raw_crop_cyx)

    stem = f"{zpath.name}_L{level}_crop_r{rmin}-{rmax}_c{cmin}-{cmax}"

    raw_tif = OUTDIR / f"{stem}_RAW_CYX_uint16.tif"
    norm_tif = OUTDIR / f"{stem}_NORMp1p99_CYX_uint16.tif"
    fig_tif = OUTDIR / f"{stem}_RAW_vs_NORMALIZED_RGB_fig.tif"

    tiff.imwrite(raw_tif, raw_crop_cyx.astype(np.uint16), imagej=True)
    tiff.imwrite(norm_tif, norm_crop_cyx.astype(np.uint16), imagej=True)

    save_comparison_figure(
        raw_crop_rgb_uint8,
        norm_crop_rgb_float,
        fig_tif,
        title=f"{zpath.name} | level {level} | crop [{rmin}:{rmax}, {cmin}:{cmax}] | RAW vs NORMALIZED"
    )

    print("\nSaved:")
    print(f"  RAW crop TIFF (uint16)        : {raw_tif}")
    print(f"  NORMALIZED crop TIFF (uint16) : {norm_tif}")
    print(f"  RAW vs NORMALIZED figure      : {fig_tif}")

    if ask_yes_no("\nExtract RAW patches from the RAW crop (per channel)?", default=True):
        patch = ask_int("Patch size", 8, 512, default=64)
        stride = ask_int("Stride", 1, 512, default=32)
        max_patches = ask_int("Max patches", 1, 200000, default=5000)

        for ch in range(raw_crop_cyx.shape[0]):
            patches = extract_patches_grid(raw_crop_cyx[ch], patch=patch, stride=stride, max_patches=max_patches)
            patches_tif = OUTDIR / f"{stem}_ch{ch}_RAW_patches_uint16_N{patches.shape[0]}_p{patch}_s{stride}.tif"
            tiff.imwrite(patches_tif, patches.astype(np.uint16), imagej=True)

            pmax = int(np.max(patches)) if patches.size else -1
            print(f"Saved RAW patches for ch{ch}: {patches_tif} | shape={patches.shape} | max_intensity={pmax}")

    print("\nDone.")


if __name__ == "__main__":
    main()