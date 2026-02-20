from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import zarr

""" Sanity check for 2-channel OME-Zarr """

OMEZARR_PATH = Path(
    r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\training_files\2d_wga_dapi\training_data\WT_CSP_NHS_ROI2_SIM_omezarr"
)


def percentile_norm01(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [p_low, p_high])
    x = x.astype(np.float32, copy=False)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        m = float(np.max(x)) if x.size else 0.0
        return (x / m) if m > 0 else np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def load_highest_res_array(zarr_path: Path) -> np.ndarray:
    """
    Loads highest-res dataset.
    Typical NGFF layout: root['0'] is the full-res array.
    """
    root = zarr.open(str(zarr_path), mode="r")

    if "0" in root:
        return np.asarray(root["0"])

    keys = list(root.array_keys())
    if keys:
        return np.asarray(root[keys[0]])

    for gk in root.group_keys():
        g = root[gk]
        ak = list(getattr(g, "array_keys", lambda: [])())
        if ak:
            return np.asarray(g[ak[0]])

    raise RuntimeError(f"Could not find any array dataset inside: {zarr_path}")


def choose_random_2d_slice(arr: np.ndarray) -> np.ndarray:
    """
    Returns a 3D array with channels + 2D spatial image:
      either (C,Y,X) or (Y,X,C)

    If arr has extra dims (T/Z), it picks random indices for those dims.
    """
    a = np.asarray(arr)

    if a.ndim < 3:
        raise ValueError(f"Expected at least 3D with channels, got shape={a.shape}")

    shape = a.shape
    small_axes = [i for i, s in enumerate(shape) if s <= 4]

    if not small_axes:
        raise ValueError(f"Cannot find a channel-like axis (<=4) in shape={shape}")

    candidate_axes = [ax for ax in small_axes if ax not in (a.ndim - 1, a.ndim - 2)]
    cax = candidate_axes[0] if candidate_axes else small_axes[0]

    idx = []
    for ax, size in enumerate(shape):
        if ax == cax:
            idx.append(slice(None))  # keep all channels
        elif ax in (a.ndim - 2, a.ndim - 1):
            idx.append(slice(None))  # keep Y,X
        else:
            idx.append(random.randrange(size))  # pick random 

    a2 = a[tuple(idx)]

    if a2.ndim != 3:
        raise ValueError(f"After slicing, expected 3D but got shape={a2.shape}")

    return a2


def split_channels(a3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract channel 0 and 1 from either (C,Y,X) or (Y,X,C).
    """
    if a3.shape[0] <= 4:  # (C,Y,X)
        if a3.shape[0] < 2:
            raise ValueError(f"Need at least 2 channels, got shape={a3.shape}")
        return a3[0], a3[1]

    if a3.shape[-1] <= 4:  # (Y,X,C)
        if a3.shape[-1] < 2:
            raise ValueError(f"Need at least 2 channels, got shape={a3.shape}")
        return a3[..., 0], a3[..., 1]

    raise ValueError(f"Cannot infer channel axis from shape={a3.shape}")


def main() -> None:
    if not OMEZARR_PATH.exists():
        raise FileNotFoundError(f"OME-Zarr not found: {OMEZARR_PATH}")

    arr = load_highest_res_array(OMEZARR_PATH)
    a3 = choose_random_2d_slice(arr)
    c0, c1 = split_channels(a3)

    c0n = percentile_norm01(c0)
    c1n = percentile_norm01(c1)

    rgb = np.zeros((*c0n.shape, 3), dtype=np.float32)
    rgb[..., 2] = c0n  # Blue
    rgb[..., 1] = c1n  # Green

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(c0n, cmap="gray")
    ax1.set_title("Channel 0 (grayscale)")
    ax1.axis("off")

    ax2.imshow(c1n, cmap="gray")
    ax2.set_title("Channel 1 (grayscale)")
    ax2.axis("off")

    ax3.imshow(rgb)
    ax3.set_title("Overlay: ch0=Blue, ch1=Green")
    ax3.axis("off")

    fig.suptitle(f"{OMEZARR_PATH.name} | raw shape={arr.shape} | shown slice shape={a3.shape}", fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

