from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import czifile
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class Paths:
    data_2d_time: Path = Path(r"E:\2D_data_time")
    data_2d_wga_dapi: Path = Path(r"E:\2D_data_WGA_DAPI_DNA")
    data_3d: Path = Path(r"E:\20220225_HADA_NADA_TADA_40min")


PATHS = Paths()

CMAP_BLUE_BLACK = LinearSegmentedColormap.from_list("blue_black", [(0, 0, 0), (0, 0, 1)])
CMAP_GREEN_BLACK = LinearSegmentedColormap.from_list("green_black", [(0, 0, 0), (0, 1, 0)])
CMAP_RED_BLACK = LinearSegmentedColormap.from_list("red_black", [(0, 0, 0), (1, 0, 0)])


def list_czi_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = sorted(folder.glob("*.czi"))
    if not files:
        raise FileNotFoundError(f"No .czi files found in: {folder}")
    return files


def pick_random_file(folder: str | Path) -> Path:
    return random.choice(list_czi_files(folder))


def read_czi_array(czi_path: str | Path) -> np.ndarray:
    """Read CZI into numpy """
    czi_path = Path(czi_path)
    with czifile.CziFile(str(czi_path)) as czi:
        arr = czi.asarray()
    return np.squeeze(arr)


def read_czi_header(czi_path: str | Path) -> tuple[str | None, tuple[int, ...] | None]:
    """Read axes + shape."""
    czi_path = Path(czi_path)
    with czifile.CziFile(str(czi_path)) as czi:
        axes = getattr(czi, "axes", None)
        shape = getattr(czi, "shape", None)
    return axes, shape


def normalize01(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8, thr: float = 0.0) -> np.ndarray:
    """Normalization for fluorescence."""
    img = img.astype(np.float32, copy=False)
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        hi = lo + 1e-6
    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if thr and thr > 0:
        out[out < thr] = 0.0
    return out


def max_project_to_2d(arr: np.ndarray) -> np.ndarray:
    """Max-project all dims except last two(for 2d)."""
    if arr.ndim == 2:
        return arr
    reduce_axes = tuple(range(arr.ndim - 2))
    return arr.max(axis=reduce_axes) if reduce_axes else arr


def find_channel_axis(arr: np.ndarray) -> int | None:
    """Locate channel axises"""
    if arr.ndim < 3:
        return None
    for ax in range(arr.ndim - 2):
        if 2 <= arr.shape[ax] <= 4:
            return ax
    return None


def get_channel_2d(arr: np.ndarray, ch_index: int, channel_axis: int | None = None) -> np.ndarray:
    """Extract channel 2D."""
    if channel_axis is None:
        channel_axis = find_channel_axis(arr)
    if channel_axis is not None and arr.shape[channel_axis] > 1:
        arr = np.take(arr, indices=ch_index, axis=channel_axis)
    return max_project_to_2d(arr)


def squeeze_axes_from_header(axes_full: str | None, shape_full: tuple[int, ...] | None) -> str:
    """Create squeezed axes string if header dims > 1."""
    if (
        isinstance(axes_full, str)
        and isinstance(shape_full, tuple)
        and len(axes_full) == len(shape_full)
    ):
        return "".join(ax for ax, n in zip(axes_full, shape_full) if n > 1)
    return ""


def move_to_czyx(arr: np.ndarray, axes_squeezed: str) -> np.ndarray:
    """Reorder  array to (C,Z,Y,X)."""
    if not axes_squeezed or not all(k in axes_squeezed for k in "CZYX"):
        raise ValueError(f"Cannot find C/Z/Y/X axes. axes_squeezed='{axes_squeezed}', arr.shape={arr.shape}")

    c_ax = axes_squeezed.index("C")
    z_ax = axes_squeezed.index("Z")
    y_ax = axes_squeezed.index("Y")
    x_ax = axes_squeezed.index("X")
    return np.moveaxis(arr, (c_ax, z_ax, y_ax, x_ax), (0, 1, 2, 3))



def show_rgb_on_black(rgb: np.ndarray, title: str) -> None:
    fig = plt.figure(facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.imshow(rgb, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    ax.axis("off")


def show_gray_or_cmap_on_black(img01: np.ndarray, title: str, cmap) -> None:
    fig = plt.figure(facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.imshow(img01, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white")
    ax.axis("off")


def preview_random_2d_time_hada_blue(folder: str | Path = PATHS.data_2d_time) -> Path:
    """2D time data: single channel HADA -> BLUE."""
    f = pick_random_file(folder)
    arr = read_czi_array(f)
    img2d = max_project_to_2d(arr)
    b = normalize01(img2d)
    rgb = np.dstack([np.zeros_like(b), np.zeros_like(b), b])
    show_rgb_on_black(rgb, f"2D_time HADA (BLUE) | {f.name}")
    plt.show(block=False)
    return f


def preview_random_2d_wga_dapi(
    folder: str | Path = PATHS.data_2d_wga_dapi,
    wga_ch: int = 0,
    dapi_ch: int = 1,
) -> Path:
    """2D WGA/DAPI: WGA -> GREEN, DAPI -> BLUE."""
    f = pick_random_file(folder)
    arr = read_czi_array(f)
    ch_ax = find_channel_axis(arr)

    wga = normalize01(get_channel_2d(arr, wga_ch, ch_ax))
    dapi = normalize01(get_channel_2d(arr, dapi_ch, ch_ax))
    rgb = np.dstack([np.zeros_like(wga), wga, dapi])

    show_rgb_on_black(rgb, f"2D WGA(GREEN) + DAPI(BLUE) | {f.name} | wga={wga_ch}, dapi={dapi_ch}")
    plt.show(block=False)
    return f


def info_and_plot_zslices(
    folder: str | Path = PATHS.data_3d,
    n_slices: int = 3,
    channel_labels: list[str] | None = None,
    p_low: float = 1.0,
    p_high: float = 99.8,
    thr: float = 0.05,
    rgb_order: tuple[int, int, int] = (2, 1, 0),  # (R,G,B) indices
):
    """
    Random 3D CZI -> print info -> plot 3 random Z slices.
    """
    f = pick_random_file(folder)

    axes_full, shape_full = read_czi_header(f)
    arr = read_czi_array(f)

    axes_squeezed = squeeze_axes_from_header(axes_full, shape_full)
    if not axes_squeezed:
        axes_squeezed = "?" * arr.ndim

    counts = {}
    if isinstance(axes_full, str) and isinstance(shape_full, tuple) and len(axes_full) == len(shape_full):
        for ax, n in zip(axes_full, shape_full):
            counts[ax] = int(n)

    C = counts.get("C", 1)
    Z = counts.get("Z", 1)
    T = counts.get("T", 1)
    S = counts.get("S", 1)
    Y = counts.get("Y", counts.get("H", 0))
    X = counts.get("X", counts.get("W", 0))

    print("\n=== Random 3D CZI info (3 random Z; BLACK background) ===")
    print("File:", f.name)
    print("Path:", str(f))
    print("Axes:", axes_full)
    print("Header shape:", shape_full)
    print("Squeezed axes:", axes_squeezed)
    print("Array shape :", arr.shape)
    print("Dtype:", arr.dtype)
    print(f"Intensity min/max: {float(arr.min())} / {float(arr.max())}")
    print(f"Channels (C): {C}")
    print(f"Z-slices  (Z): {Z}")
    print(f"Timepoints(T): {T}")
    print(f"Scenes   (S): {S}")
    if Y and X:
        print(f"Image size: {Y} x {X} (Y x X)")
    print("All dim counts:", counts)

    # defaults
    if channel_labels is None:
        channel_labels = ["HADA", "NADA", "TADA"]
    per_channel_cmaps = [CMAP_BLUE_BLACK, CMAP_GREEN_BLACK, CMAP_RED_BLACK]

    arr_czyx = move_to_czyx(arr, axes_squeezed)
    C2, Z2, _, _ = arr_czyx.shape

    # pick random Z 
    n = min(n_slices, Z2)
    z_idx = sorted(random.sample(range(Z2), k=n))

    r_i, g_i, b_i = rgb_order#RGB
    for idx in (r_i, g_i, b_i):
        if idx < 0 or idx >= C2:
            raise ValueError(f"rgb_order {rgb_order} invalid for C={C2}")

    for zi in z_idx:
        # individual channels
        for ci in range(C2):
            label = channel_labels[ci] if ci < len(channel_labels) else f"Ch{ci}"
            cmap = per_channel_cmaps[min(ci, 2)]
            img01 = normalize01(arr_czyx[ci, zi], p_low=p_low, p_high=p_high, thr=thr)
            show_gray_or_cmap_on_black(img01, f"{f.name} | {label} | Z={zi}/{Z2-1}", cmap)

        # RGB 
        r = normalize01(arr_czyx[r_i, zi], p_low=p_low, p_high=p_high, thr=thr)
        g = normalize01(arr_czyx[g_i, zi], p_low=p_low, p_high=p_high, thr=thr)
        b = normalize01(arr_czyx[b_i, zi], p_low=p_low, p_high=p_high, thr=thr)
        rgb = np.dstack([r, g, b])
        show_rgb_on_black(rgb, f"{f.name} | RGB composite (R={r_i},G={g_i},B={b_i}) | Z={zi}/{Z2-1}")

    plt.show()

    return {
        "file": f.name,
        "path": str(f),
        "axes": axes_full,
        "header_shape": shape_full,
        "axes_squeezed": axes_squeezed,
        "array_shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "C": C2,
        "Z": Z2,
        "random_z": z_idx,
        "thr": thr,
        "rgb_order": rgb_order,
    }



def main() -> None:
    preview_random_2d_time_hada_blue()
    preview_random_2d_wga_dapi(wga_ch=1, dapi_ch=0)
    info = info_and_plot_zslices()
    print(info)
    plt.show()


if __name__ == "__main__":
    main()



