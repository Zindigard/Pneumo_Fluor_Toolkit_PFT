from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from PFT.core_prog_parts.decoder_omezar import ome_zarr_to_n2v_2d_stack
from PFT.core_prog_parts.n2v_denoising import load_n2v_model


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root_2d_time(root: Path) -> Path:
    return root / "results" / "training_files" / "2d_time"


def list_all_omezarr_under(folder: Path) -> List[Path]:
    candidates = []
    for p in folder.rglob("*"):
        if p.is_dir() and (
            p.name.lower().endswith(".ome.zarr")
            or p.name.lower().endswith("_omezarr")
        ):
            candidates.append(p)
    return sorted(set(candidates))


def choose_from_list(title: str, items: List[str]) -> int:
    print(f"\n{title}")
    for i, s in enumerate(items, 1):
        print(f"  {i}) {s}")
    while True:
        ans = input(f"Enter 1..{len(items)} (or 'q'): ").strip().lower()
        if ans == "q":
            raise SystemExit(0)
        if ans.isdigit():
            k = int(ans)
            if 1 <= k <= len(items):
                return k - 1
        print("Invalid input.")


# -------------------------------------------------------
# Load RAW (no normalization)
# -------------------------------------------------------

def load_first_frame_ch0_raw(sample_zarr_dir: Path) -> np.ndarray:
    stack = ome_zarr_to_n2v_2d_stack(
        sample_zarr_dir,
        channel=0,
        time=None,
        z=None,
        normalize=None,
    )
    stack = np.asarray(stack)
    return stack[0, :, :, 0]  # uint16


# -------------------------------------------------------
# Plot RAW32 vs RAW32
# -------------------------------------------------------

def show_raw32_vs_denoised(raw32: np.ndarray, den32: np.ndarray, title: str):
    diff = den32 - raw32

    clip = float(np.percentile(np.abs(diff), 99.5))
    clip = max(clip, 1.0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    ax[0].imshow(raw32, cmap="gray")
    ax[0].set_title("Input RAW32")
    ax[0].axis("off")

    ax[1].imshow(den32, cmap="gray")
    ax[1].set_title("Denoised RAW32")
    ax[1].axis("off")

    im = ax[2].imshow(diff, cmap="seismic", vmin=-clip, vmax=clip)
    ax[2].set_title(f"Difference\nclip=±{clip:.1f}")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    plt.show()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main() -> int:
    root = repo_root_from_this_file()
    ds_root = dataset_root_2d_time(root)

    samples = list_all_omezarr_under(ds_root)
    if not samples:
        print("[ERROR] No samples found.")
        return 1

    labels = [str(p.relative_to(ds_root)) for p in samples]
    idx = choose_from_list("Choose sample to denoise (ch0):", labels)
    sample = samples[idx]

    # 1) Load RAW uint16
    raw_u16 = load_first_frame_ch0_raw(sample)

    # 2) Convert to RAW32 (no normalization)
    raw32 = raw_u16.astype(np.float32)

    # 3) Load model manually
    model = load_n2v_model("n2v_2d_time_blue_p64x64")

    # 4) Predict directly on RAW32
    inp = raw32[np.newaxis, ..., np.newaxis]  # (1, Y, X, 1)
    den = model.predict(inp)
    den32 = den[0, ..., 0].astype(np.float32)

    # 5) Print stats
    diff = den32 - raw32
    print("\n========== RAW32 DIFF STATS ==========")
    print("mean:", float(diff.mean()))
    print("std :", float(diff.std()))
    print("min :", float(diff.min()))
    print("max :", float(diff.max()))
    print("p99(|diff|):", float(np.percentile(np.abs(diff), 99)))
    print("Input std:", float(raw32.std()))
    print("Denoised std:", float(den32.std()))
    print("======================================\n")

    # 6) Show
    show_raw32_vs_denoised(raw32, den32, title=sample.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())