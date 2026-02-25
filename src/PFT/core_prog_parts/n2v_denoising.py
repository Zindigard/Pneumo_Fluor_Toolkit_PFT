from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from n2v.models import N2V

"""Helper functions to load N2V models for 2D datasets and apply them for denoising."""

#  Model
MODEL_NAMES = {
    ("2d_time", "blue"): "n2v_2d_time_blue_p64x64",
    ("2d_wga_dapi", "blue_DAPI"): "n2v_2d_wga_dapi_blue_DAPI_p64x64",
    ("2d_wga_dapi", "green_WGA"): "n2v_2d_wga_dapi_green_WGA_p64x64",
}



_MODEL_CACHE: Dict[Tuple[str, str], N2V] = {}


def project_root_from_this_file() -> Path:
    """
    File location:
     
    """
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    return project_root_from_this_file() / "models"


def get_model(dataset: str, key: str) -> N2V:
    cache_key = (dataset, key)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_dir = models_dir()
    model_name = MODEL_NAMES[(dataset, key)]

    # Load existing model from disk
    model = N2V(config=None, name=model_name, basedir=str(model_dir))
    print(f"[N2V] Loading model: {model_name}")
    print(f"[N2V] Basedir: {model_dir}")
    print(f"[N2V] Exists: {(model_dir / model_name).exists()}")
    print(f"[N2V] Has weights_best: {((model_dir / model_name) / 'weights_best.weights.h5').exists()}")
    _MODEL_CACHE[cache_key] = model
    return model

def percentile_norm01(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Percentile normalize to [0,1] float32."""
    x = np.asarray(x)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    lo, hi = np.percentile(x, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Fallback: min/max
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)

    y = (x.astype(np.float32) - float(lo)) / (float(hi) - float(lo))
    return np.clip(y, 0.0, 1.0)


def normalize_yxc_for_n2v(img_yxc: np.ndarray) -> np.ndarray:
    """
    percentile per channel -> float32 in [0,1].
    """
    img_yxc = np.asarray(img_yxc)
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected YXC image, got shape {img_yxc.shape}")

    c = img_yxc.shape[-1]
    out = np.empty(img_yxc.shape, dtype=np.float32)
    for ch in range(c):
        out[..., ch] = percentile_norm01(img_yxc[..., ch])
    return out


def predict_single_channel(img_yx: np.ndarray, model: N2V) -> np.ndarray:
    """
    img_yx: (Y, X) or (Y, X, 1)
    returns: (Y, X)
    """
    if img_yx.ndim == 2:
        x = img_yx[..., np.newaxis]
    elif img_yx.ndim == 3 and img_yx.shape[-1] == 1:
        x = img_yx
    else:
        raise ValueError(f"Expected single-channel (Y,X) or (Y,X,1), got {img_yx.shape}")

    y = model.predict(x, axes="YXC")  # (Y,X,1)
    return np.asarray(y[..., 0], dtype=np.float32)  # (Y,X)


def denoise_2d_for_cellpose(img: np.ndarray) -> np.ndarray:
    """
    Denoise 2D datasets and return YXC.

    Accepted input:
      - (Y,X)
      - (Y,X,1)
      - (Y,X,2)

    Output:
      - (Y,X,1) for single-channel input
      - (Y,X,2) for two-channel input
    """
    img = np.asarray(img)

    # (Y,X) -> 2d_time single channel
    if img.ndim == 2:
        dataset = "2d_time"
        model = get_model(dataset, "blue")

        img_n = percentile_norm01(img)
        den = predict_single_channel(img_n, model)     # (Y,X)
        return den[..., np.newaxis]                    # (Y,X,1)

    # (Y,X,C)
    if img.ndim == 3:
        # (Y,X,1)
        if img.shape[-1] == 1:
            dataset = "2d_time"
            model = get_model(dataset, "blue")

            ch0 = percentile_norm01(img[..., 0])
            den = predict_single_channel(ch0, model)
            return den[..., np.newaxis]                # (Y,X,1)

        # (Y,X,2) -> 2d_wga_dapi
        if img.shape[-1] == 2:
            dataset = "2d_wga_dapi"
            model_dapi = get_model(dataset, "blue_DAPI")
            model_wga = get_model(dataset, "green_WGA")

            ch0 = percentile_norm01(img[..., 0])
            ch1 = percentile_norm01(img[..., 1])

            den0 = predict_single_channel(ch0, model_dapi)  # DAPI
            den1 = predict_single_channel(ch1, model_wga)   # WGA
            return np.stack([den0, den1], axis=-1)           # (Y,X,2)

    raise ValueError(
        f"Unsupported input shape {img.shape}. Expected (Y,X), (Y,X,1) or (Y,X,2)."
    )
