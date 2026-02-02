import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from czifile import imread
except Exception as e:
    raise RuntimeError("Missing dependency: czifile. Install with `pip install czifile`.") from e

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from n2v.models import N2V, N2VConfig
    N2V_AVAILABLE = True
except Exception as _n2v_err:
    N2V_AVAILABLE = False
    def _n2v_missing(*args, **kwargs):
        raise RuntimeError(
            "Noise2Void (n2v) is not installed or failed to import. "
            "Install compatible versions, e.g.: "
            "pip install 'n2v==0.3.3' 'csbdeep>=0.7,<0.8' 'tensorflow>=2.10,<2.16'"
        ) from _n2v_err
    N2V = _n2v_missing
    N2VConfig = _n2v_missing

PATCH_SIZE       = (64, 64) 
N_PATCHES        = 512
VAL_SPLIT        = 0.10
TRAIN_EPOCHS     = 20
STEPS_PER_EPOCH  = 100
BATCH_SIZE       = 16
LEARNING_RATE    = 1e-3
REDUCE_LR        = {'patience': 3, 'factor': 0.5}
BASE_MODEL_DIR   = './models'   # where N2V stores its training logs

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x, np.float32) if mx <= mn else (x - mn) / (mx - mn)

def _psnr_ssim(a: np.ndarray, b: np.ndarray) -> tuple:
    a, b = normalize(a), normalize(b)
    return (peak_signal_noise_ratio(a, b, data_range=1.0),
            structural_similarity(a, b, data_range=1.0))

def extract_czyx_channels_from_czi(raw) -> np.ndarray:
    """
    Accepts czifile.imread(...) output with shapes like:
      (T,C,Z,Y,X), (C,Z,Y,X), (Z,C,Y,X), (C,Y,X), (Y,X).
    Returns (C, Y, X) float32 in [0,1] (Z=0, T=0 if present).
    """
    arr = np.asarray(raw)
    arr = np.squeeze(arr)

    if arr.ndim == 5:          # (T,C,Z,Y,X) or (T,Z,C,Y,X)
        arr = arr[0]           # T=0 -> (C,Z,Y,X) or (Z,C,Y,X)
        if arr.shape[0] < 3 and arr.shape[1] >= 3:
            arr = np.moveaxis(arr, 0, 1)   # (Z,C,...) -> (C,Z,...)
        if arr.shape[1] > 1:
            arr = arr[:, 0]                # Z=0 -> (C,Y,X)

    elif arr.ndim == 4:        # (C,Z,Y,X) or (Z,C,Y,X)
        if arr.shape[0] < 3 and arr.shape[1] >= 3:
            arr = np.moveaxis(arr, 0, 1)   # (Z,C,...) -> (C,Z,...)
        if arr.shape[1] > 1:
            arr = arr[:, 0]                # Z=0 -> (C,Y,X)

    elif arr.ndim == 3:
        # Could be (C,Y,X) or (Y,X,C). If last dim looks like channels (<=4) swap.
        if arr.shape[-1] <= 4 and arr.shape[0] > 8 and arr.shape[1] > 8:
            arr = np.moveaxis(arr, -1, 0)  # (Y,X,C) -> (C,Y,X)

    elif arr.ndim == 2:
        arr = arr[None, :, :]               # -> (1,Y,X)

    else:
        raise ValueError(f"Unexpected CZI array shape: {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Expected (C,Y,X), got {arr.shape}")

    return normalize(arr.astype(np.float32))

def _pad_to_multiple(img2d: np.ndarray, m: int = 4):
    """Pad reflectively so H,W are multiples of m; return padded image and crop slices."""
    H, W = img2d.shape
    Hn = ((H + m - 1) // m) * m
    Wn = ((W + m - 1) // m) * m
    pad_y = Hn - H
    pad_x = Wn - W
    top = pad_y // 2; bottom = pad_y - top
    left = pad_x // 2; right = pad_x - left
    if pad_y or pad_x:
        pad_img = np.pad(img2d, ((top, bottom), (left, right)), mode='reflect')
        crop = (slice(top, top + H), slice(left, left + W))
        return pad_img, crop
    return img2d, (slice(0, H), slice(0, W))

def train_n2v_for_channel(img2d: np.ndarray, model_name: str,
                          basedir: str | Path = BASE_MODEL_DIR) -> np.ndarray:
    """
    Train per-image Noise2Void on a single 2D channel.
    img2d: float32 (Y,X) in [0,1]
    Returns denoised float32 (Y,X) in [0,1]
    """
    if not N2V_AVAILABLE:
        raise RuntimeError("Noise2Void (n2v) is not available. See the install hint above.")

    os.makedirs(basedir, exist_ok=True)

    rng = np.random.default_rng(42)
    H, W = img2d.shape
    ph, pw = PATCH_SIZE
    patches = []
    for _ in range(N_PATCHES):
        top  = rng.integers(0, max(1, H - ph))
        left = rng.integers(0, max(1, W - pw))
        patches.append(img2d[top:top+ph, left:left+pw][..., None])
    patches = np.stack(patches).astype(np.float32)  # (N, ph, pw, 1)

    n_val = int(len(patches) * VAL_SPLIT)
    val_X, tr_X = patches[:n_val], patches[n_val:]

    cfg = N2VConfig(
        X=patches, axes='YXC',
        unet_kern_size=3,
        train_steps_per_epoch=STEPS_PER_EPOCH,
        train_epochs=TRAIN_EPOCHS, train_loss='mse',
        batch_norm=True, train_batch_size=BATCH_SIZE,
        n2v_perc_pix=0.198, n2v_patch_shape=PATCH_SIZE,
        train_learning_rate=LEARNING_RATE,
        train_reduce_lr=REDUCE_LR
    )
    n2v = N2V(config=cfg, name=model_name, basedir=str(basedir))
    n2v.train(tr_X, val_X, epochs=TRAIN_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

    pad_img, crop = _pad_to_multiple(img2d, m=4)
    pred_full = n2v.predict(pad_img, axes='YX').astype(np.float32)
    pred = pred_full[crop]
    return normalize(pred)

def run_n2v_denoising(image_list: list[str] | None = None,
                      csv_path: str | Path = "psnr_ssim_3ch.csv") -> pd.DataFrame:
    """
    For each CZI in image_list (or auto-discovers *.czi), create or load:
        <stem>_denoised_R.npy, _G.npy, _B.npy
    and write PSNR/SSIM per channel to csv_path.
    Returns the DataFrame.
    """
    if image_list is None:
        image_list = list(globals().get('image_files') or sorted(glob("*.czi")))
    if not image_list:
        raise RuntimeError("No input images. Define `image_files=[...]` or place *.czi in the current folder.")

    rows = []
    for i, img_path in enumerate(image_list, start=1):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        raw = imread(img_path)
        cyx = extract_czyx_channels_from_czi(raw)  # (C,Y,X) in [0,1]

        C = cyx.shape[0]
        b_idx = 0 if C >= 1 else 0
        g_idx = 1 if C >= 2 else 0
        r_idx = 2 if C >= 3 else C - 1

        R = normalize(cyx[r_idx]); G = normalize(cyx[g_idx]); B = normalize(cyx[b_idx])

        stem = str(Path(img_path).with_suffix(''))
        fR = f"{stem}_denoised_R.npy"
        fG = f"{stem}_denoised_G.npy"
        fB = f"{stem}_denoised_B.npy"

        # Train if cache missing; else load
        if not (os.path.exists(fR) and os.path.exists(fG) and os.path.exists(fB)):
            print(f"\n[TRAIN N2V] {Path(img_path).name}")
            denR = train_n2v_for_channel(R, f"N2V_{i}_R"); np.save(fR, denR)
            denG = train_n2v_for_channel(G, f"N2V_{i}_G"); np.save(fG, denG)
            denB = train_n2v_for_channel(B, f"N2V_{i}_B"); np.save(fB, denB)
        else:
            print(f"[SKIP] Found denoised cache for {Path(img_path).name}")
            denR = np.load(fR).astype(np.float32)
            denG = np.load(fG).astype(np.float32)
            denB = np.load(fB).astype(np.float32)

        for ch_name, raw_ch, den_ch in [('R', R, denR), ('G', G, denG), ('B', B, denB)]:
            pscore, sscore = _psnr_ssim(raw_ch, den_ch)
            rows.append({'Image': Path(img_path).name,
                         'Channel': ch_name, 'PSNR': pscore, 'SSIM': sscore})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df