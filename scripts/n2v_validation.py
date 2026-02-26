from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PFT.core_prog_parts.decoder_omezar import ome_zarr_to_n2v_2d_stack
from PFT.core_prog_parts.n2v_denoising import load_n2v_model

def find_repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "models").exists() and (parent / "results").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def dataset_root_2d_wga_dapi(root: Path) -> Path:
    return root / "results" / "training_files" / "2d_wga_dapi"


def is_omezarr_store(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / ".zgroup").exists() or (p / "zarr.json").exists() or (p / ".zattrs").exists()


def list_all_omezarr_under(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    direct = [p for p in folder.iterdir() if is_omezarr_store(p)]
    if direct:
        return sorted(direct)

    nested: List[Path] = []
    for sub in [p for p in folder.iterdir() if p.is_dir()]:
        nested.extend([p for p in sub.iterdir() if is_omezarr_store(p)])
    return sorted(nested)


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


def load_first_frame_wga_dapi_raw32(sample_zarr_dir: Path) -> np.ndarray:
    """
    Returns (Y,X,2) float32:
      C0 = DAPI (blue)
      C1 = WGA  (green)
    """
    s0 = ome_zarr_to_n2v_2d_stack(sample_zarr_dir, channel=0, time=None, z=None, normalize=None)
    s1 = ome_zarr_to_n2v_2d_stack(sample_zarr_dir, channel=1, time=None, z=None, normalize=None)

    s0 = np.asarray(s0)  # (N,Y,X,1) uint16
    s1 = np.asarray(s1)

    f0 = s0[0, :, :, 0].astype(np.float32, copy=False)
    f1 = s1[0, :, :, 0].astype(np.float32, copy=False)

    return np.stack([f0, f1], axis=-1)  # (Y,X,2)

def _display_rgb_from_raw(raw_yxc: np.ndarray) -> np.ndarray:
    """
    Create RGB image from RAW32 (Y,X,2) WITHOUT any scaling or normalization.

    Mapping:
      - blue  = raw[...,0]  (DAPI)
      - green = raw[...,1]  (WGA)

    Data remains in original RAW32 intensity space.
    """
    raw_yxc = np.asarray(raw_yxc, dtype=np.float32)
    y, x, c = raw_yxc.shape
    assert c == 2, f"Expected 2 channels, got {c}"

    rgb = np.zeros((y, x, 3), dtype=np.float32)
    rgb[..., 2] = raw_yxc[..., 0]  # Blue (DAPI)
    rgb[..., 1] = raw_yxc[..., 1]  # Green (WGA)

    return rgb


def show_before_after_joint(raw_yxc, den_yxc, title):

    dapi_raw = raw_yxc[..., 0]
    wga_raw  = raw_yxc[..., 1]

    p_low = 1
    p_high = 99.8

    d_lo, d_hi = np.percentile(dapi_raw, (p_low, p_high))
    w_lo, w_hi = np.percentile(wga_raw, (p_low, p_high))

    def normalize(channel, lo, hi):
        norm = (channel - lo) / (hi - lo + 1e-8)
        return np.clip(norm, 0, 1)

    def make_rgb(arr):
        rgb = np.zeros((*arr.shape[:2], 3), dtype=np.float32)
        rgb[..., 2] = normalize(arr[..., 0], d_lo, d_hi) 
        rgb[..., 1] = normalize(arr[..., 1], w_lo, w_hi)  
        return rgb

    before = make_rgb(raw_yxc)
    after  = make_rgb(den_yxc)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax[0].imshow(before)
    ax[0].set_title("Before (normalized for display)")
    ax[0].axis("off")

    ax[1].imshow(after)
    ax[1].set_title("After N2V (normalized for display)")
    ax[1].axis("off")

    fig.suptitle(title)
    plt.show()

def main() -> int:
    MODEL_NAME = "n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw"

    root = find_repo_root()
    ds_root = dataset_root_2d_wga_dapi(root)

    samples = list_all_omezarr_under(ds_root)
    if not samples:
        print(f"[ERROR] No OME-Zarr samples found under: {ds_root}")
        return 1

    labels = [str(p.relative_to(ds_root)) for p in samples]
    idx = choose_from_list("Choose sample to validate (WGA+DAPI joint, first frame):", labels)
    sample = samples[idx]

    raw_yxc = load_first_frame_wga_dapi_raw32(sample)

    model = load_n2v_model(MODEL_NAME)

    try:
        den_yxc = model.predict(raw_yxc, axes="YXC")
    except TypeError:
        den_yxc = model.predict(raw_yxc)
    den_yxc = np.asarray(den_yxc, dtype=np.float32)  # (Y,X,2)

    print("\n========== INPUT/OUTPUT STATS (RAW32) ==========")
    for ch, name in [(0, "DAPI (blue)"), (1, "WGA (green)")]:
        r = raw_yxc[..., ch]
        d = den_yxc[..., ch]
        diff = d - r
        print(f"\nChannel {ch} — {name}")
        print("  raw  : min/max/mean/std =", float(r.min()), float(r.max()), float(r.mean()), float(r.std()))
        print("  den  : min/max/mean/std =", float(d.min()), float(d.max()), float(d.mean()), float(d.std()))
        print("  diff : mean/std/p99(|diff|) =", float(diff.mean()), float(diff.std()), float(np.percentile(np.abs(diff), 99)))
    print("\n(Visualization uses percentile scaling ONLY for display; arrays stay RAW32.)")
    print("===============================================\n")

    show_before_after_joint(raw_yxc, den_yxc, title=sample.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())