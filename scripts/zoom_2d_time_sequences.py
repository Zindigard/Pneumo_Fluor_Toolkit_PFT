from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

BASE_DIR = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\2d_time")
TIMEPOINTS = (5, 20, 40, 120)
PREVIEW_NAME = "preview.png"  
SCALEBAR_UM = 4.0              
OUT_DIR = BASE_DIR / "zoomed"


def sanitize_for_windows(name: str) -> str:
    # Remove characters in windows paths
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def parse_timepoint(folder_name: str) -> int | None:
    """
    Extract time in minutes from folder name
    """
    m = re.search(r"_(\d+)min_", folder_name, flags=re.IGNORECASE)
    if not m:
        return None
    t = int(m.group(1))
    return t if t in TIMEPOINTS else None


def make_sequence_key(folder_name: str) -> str:
    """
    Group sequences by 5/20/40/120, they end up in the same group.
    """
    return re.sub(r"_(?:5|20|40|120)min_", "_TIME_", folder_name, flags=re.IGNORECASE)


def find_sequences(base_dir: Path) -> list[dict]:
    """
    Look for folders in base_dir that contain preview.png.
    Group them into sequences with all TIMEPOINTS.
    """
    seq_map: dict[str, dict[int, Path]] = {}

    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        t = parse_timepoint(child.name)
        if t is None:
            continue

        preview = child / PREVIEW_NAME
        if not preview.exists():
            continue

        key = make_sequence_key(child.name)
        seq_map.setdefault(key, {})
        seq_map[key][t] = preview

    sequences = []
    for key, items in seq_map.items():
        if all(tp in items for tp in TIMEPOINTS):
            sequences.append({"key": key, "items": items})

    sequences.sort(key=lambda d: d["key"].lower())
    return sequences


def read_pixel_size_um_from_metadata(preview_png: Path) -> float | None:
    """
    Try to read pixel_size_um_x from metadata.txt in the same folder as preview.png.
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


def load_png(path: Path) -> np.ndarray:
    img = plt.imread(path)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

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


def select_roi_interactive(img: np.ndarray, title: str) -> ROI:
    """
    Opens a window, lets the user draw a rectangle.

    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(img, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    ax.axis("off")

    roi = {"x1": None, "x2": None, "y1": None, "y2": None}

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        roi["x1"], roi["x2"] = sorted([int(round(x1)), int(round(x2))])
        roi["y1"], roi["y2"] = sorted([int(round(y1)), int(round(y2))])

    rect = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    print("\nROI selection:")
    print(" - Draw rectangle with mouse")
    print(" - Press ENTER to accept")
    print(" - Close the window to cancel\n")

    accepted = {"ok": False}

    def on_key(event):
        if event.key == "enter":
            if None not in (roi["x1"], roi["x2"], roi["y1"], roi["y2"]):
                accepted["ok"] = True
                plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not accepted["ok"]:
        raise RuntimeError("ROI selection cancelled or not confirmed (press ENTER).")

    return ROI(roi["x1"], roi["x2"], roi["y1"], roi["y2"])


def save_crop_with_scalebar(
    crop_rgb: np.ndarray,
    out_path: Path,
    title: str,
    um_per_px: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(crop_rgb, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    ax.axis("off")

    add_scalebar(ax, um_per_px=um_per_px, length_um=SCALEBAR_UM)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base dir not found: {BASE_DIR}")

    sequences = find_sequences(BASE_DIR)
    if not sequences:
        print(f"No complete sequences found in: {BASE_DIR}")
        print(f"Expected folders containing {PREVIEW_NAME} with _5min_/_20min_/_40min_/_120min_.")
        return

    print("\nFound sequences:")
    for i, seq in enumerate(sequences):
        example_name = seq["key"].replace("_TIME_", "_5min_")
        print(f"[{i}] {example_name}")

    idx_str = input("\nChoose sequence index (empty=0): ").strip()
    idx = int(idx_str) if idx_str else 0
    if idx < 0 or idx >= len(sequences):
        raise ValueError("Invalid sequence index")

    seq = sequences[idx]
    items = seq["items"]  

    um_per_px = read_pixel_size_um_from_metadata(items[5])
    if um_per_px is None:
        s = input("Could not read pixel_size_um_x from metadata.txt. Enter µm per pixel (e.g. 0.03224): ").strip()
        um_per_px = float(s)

    # Load first (5 min) and choose ROI
    img5 = load_png(items[5])
    roi = select_roi_interactive(img5, title=f"Select ROI on 5min: {items[5].parent.name}")

    seq_name = sanitize_for_windows(seq["key"].replace("_TIME_", ""))
    out_seq_dir = OUT_DIR / seq_name
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    # Apply ROI to all timepoints
    for t in TIMEPOINTS:
        png_path = items[t]
        img = load_png(png_path)

        h, w = img.shape[0], img.shape[1]
        x1 = max(0, min(roi.x1, w - 1))
        x2 = max(0, min(roi.x2, w))
        y1 = max(0, min(roi.y1, h - 1))
        y2 = max(0, min(roi.y2, h))

        crop = img[y1:y2, x1:x2, :]

        out_png = out_seq_dir / f"{t}min_zoom.png"
        title = f"{t} min | {png_path.parent.name}"
        save_crop_with_scalebar(crop, out_png, title=title, um_per_px=um_per_px)

        print(f"Saved: {out_png}")

    print(f"\nDone. Zoom crops saved in:\n{out_seq_dir}\n")


if __name__ == "__main__":
    main()