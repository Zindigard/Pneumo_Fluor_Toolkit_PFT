from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

"""Script to interactively select ROIs in 2D WGA+DAPI previews, crop them and save the results for visualization."""

BASE_DIR = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\2d_wga_dapi")
PREVIEW_NAME = "preview.png"     
OUT_DIR = BASE_DIR / "zoomed"
SCALEBAR_UM = 4.0                


def sanitize_for_windows(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def list_preview_pngs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    previews: list[Path] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        p = child / PREVIEW_NAME
        if p.exists():
            previews.append(p)

    if not previews:
        raise FileNotFoundError(f"No {PREVIEW_NAME} found under: {base_dir}")
    return previews


def read_um_per_px_from_metadata(preview_png: Path) -> float | None:
    """
    Try to read pixel_size_um_x from metadata.txt next to preview.png.

    """
    meta_txt = preview_png.parent / "metadata.txt"
    if not meta_txt.exists():
        return None

    txt = meta_txt.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"pixel_size_um_x=([0-9eE+\-\.]+)", txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def load_png_rgb01(path: Path) -> np.ndarray:
    """Load PNG as float RGB in [0,1], shape (Y,X,3)."""
    img = plt.imread(path)

    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {img.shape}")

    return img


def add_scalebar(ax, um_per_px: float, length_um: float = SCALEBAR_UM) -> None:
    if um_per_px <= 0:
        return
    length_px = length_um / um_per_px
    fontprops = fm.FontProperties(size=10)

    bar = AnchoredSizeBar(
        ax.transData,
        size=length_px,
        label=f"{length_um:g} µm",
        loc="lower right",
        pad=0.4,
        color="white",
        frameon=False,
        size_vertical=max(1.0, length_px * 0.02),
        fontproperties=fontprops,
    )
    ax.add_artist(bar)


@dataclass
class ROI:
    x1: int
    x2: int
    y1: int
    y2: int


def select_roi_on_rgb(rgb01: np.ndarray, title: str) -> ROI:
    """
    Select ROI on RGB preview.

    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(rgb01, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    ax.axis("off")

    roi = {"x1": None, "x2": None, "y1": None, "y2": None}
    accepted = {"ok": False}

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        roi["x1"], roi["x2"] = sorted([int(round(x1)), int(round(x2))])
        roi["y1"], roi["y2"] = sorted([int(round(y1)), int(round(y2))])

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=False,          
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    print("\nROI selection:")
    print(" - Draw rectangle with mouse")
    print(" - Click once inside the window so it has focus")
    print(" - Press ENTER (or RETURN) to accept ROI")
    print(" - Close window to cancel\n")

    def on_key(event):
        if event.key in ("enter", "return"):
            if None not in (roi["x1"], roi["x2"], roi["y1"], roi["y2"]):
                accepted["ok"] = True
                plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig._roi_selector = selector

    plt.show()

    if not accepted["ok"]:
        raise RuntimeError("ROI selection cancelled or not confirmed (press ENTER/RETURN).")

    return ROI(roi["x1"], roi["x2"], roi["y1"], roi["y2"])
00


def crop_roi(rgb01: np.ndarray, roi: ROI) -> np.ndarray:
    """Crop ROI with bounds checking."""
    h, w = rgb01.shape[0], rgb01.shape[1]
    x1 = max(0, min(roi.x1, w - 1))
    x2 = max(0, min(roi.x2, w))
    y1 = max(0, min(roi.y1, h - 1))
    y2 = max(0, min(roi.y2, h))
    return rgb01[y1:y2, x1:x2, :]


def save_crop_with_scalebar(out_path: Path, rgb_crop: np.ndarray, title: str, um_per_px: float, dpi: int = 200) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(rgb_crop, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    ax.axis("off")

    add_scalebar(ax, um_per_px=um_per_px, length_um=SCALEBAR_UM)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    previews = list_preview_pngs(BASE_DIR)

    print("\nFound WGA+DAPI previews:")
    for i, p in enumerate(previews):
        print(f"[{i:03d}] {p.parent.name}")

    s = input("\nChoose file index (empty=0): ").strip()
    idx = int(s) if s else 0
    if idx < 0 or idx >= len(previews):
        raise ValueError("Invalid index.")

    preview_png = previews[idx]
    rgb = load_png_rgb01(preview_png)

    um_per_px = read_um_per_px_from_metadata(preview_png)
    if um_per_px is None:
        s = input("Could not read pixel_size_um_x from metadata.txt. Enter µm per pixel (e.g. 0.03224): ").strip()
        um_per_px = float(s)
    if um_per_px <= 0:
        raise ValueError("um_per_px must be > 0 for scalebar.")

    roi = select_roi_on_rgb(rgb, title=f"Select ROI (RGB) | {preview_png.parent.name}")
    crop = crop_roi(rgb, roi)

    folder_name = sanitize_for_windows(preview_png.parent.name)
    out_dir = OUT_DIR / folder_name
    out_path = out_dir / "zoom.png"

    save_crop_with_scalebar(
        out_path=out_path,
        rgb_crop=crop,
        title=f"Zoom | {preview_png.parent.name}",
        um_per_px=um_per_px,
    )

    print(f"\nSaved zoom crop to:\n{out_path}\n")


if __name__ == "__main__":
    main()
