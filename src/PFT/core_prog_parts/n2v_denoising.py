from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from n2v.models import N2V
"""
N2V denoising helpers.

"""


MODEL_NAMES = {
    ("2d_time", "blue"): "n2v_2d_time_blue_p64x64",

    ("2d_wga_dapi", "blue_DAPI"): "n2v_2d_wga_dapi_blue_DAPI_p64x64",
    ("2d_wga_dapi", "green_WGA"): "n2v_2d_wga_dapi_green_WGA_p64x64",

    ("2d_wga_dapi", "JOINT_DAPI+WGA_raw"): "n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw",
}

_MODEL_CACHE: Dict[Tuple[str, str], N2V] = {}
_MODEL_CACHE_BY_NAME: Dict[str, N2V] = {}


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Try to find the repo root robustly by walking upwards.
    We stop at the first parent that contains 'models' and 'results'.
    """
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "models").exists() and (parent / "results").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    return _find_repo_root() / "models"


def get_model(dataset: str, key: str) -> N2V:
    """
    Load by (dataset, key) mapping.
    """
    cache_key = (dataset, key)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_dir = models_dir()
    model_name = MODEL_NAMES[(dataset, key)]

    model = N2V(config=None, name=model_name, basedir=str(model_dir))
    print(f"[N2V] Loading model: {model_name}")
    print(f"[N2V] Basedir: {model_dir}")
    print(f"[N2V] Exists: {(model_dir / model_name).exists()}")
    print(f"[N2V] Has weights_best: {((model_dir / model_name) / 'weights_best.h5').exists()}")

    _MODEL_CACHE[cache_key] = model
    return model


def load_n2v_model(model_name: str) -> N2V:
    """
    Load by explicit model name (folder under /models).
    Used by validation/denoising scripts.
    """
    if model_name in _MODEL_CACHE_BY_NAME:
        return _MODEL_CACHE_BY_NAME[model_name]

    model_dir = models_dir()
    model = N2V(config=None, name=model_name, basedir=str(model_dir))

    print(f"[N2V] Loading model: {model_name}")
    print(f"[N2V] Basedir: {model_dir}")
    print(f"[N2V] Exists: {(model_dir / model_name).exists()}")
    print(f"[N2V] Has weights_best: {((model_dir / model_name) / 'weights_best.h5').exists()}")

    _MODEL_CACHE_BY_NAME[model_name] = model
    return model


def predict_yxc(img_yxc: np.ndarray, model: N2V) -> np.ndarray:
    """
    img_yxc: (Y, X, C) float32 (RAW range allowed)
    returns: (Y, X, C) float32

    We intentionally DO NOT add a batch dimension here because we pass a single
    image with axes="YXC". Adding a batch while still using "YXC" can lead to
    axis reordering and broadcasting errors in some n2v versions.
    """
    img_yxc = np.asarray(img_yxc, dtype=np.float32)
    if img_yxc.ndim != 3:
        raise ValueError(f"Expected (Y,X,C), got {img_yxc.shape}")

    # Some n2v versions need 'axes'
    try:
        y = model.predict(img_yxc, axes="YXC")
    except TypeError:
        y = model.predict(img_yxc)

    y = np.asarray(y, dtype=np.float32)
    if y.shape != img_yxc.shape:
        raise ValueError(f"Unexpected prediction shape {y.shape}, expected {img_yxc.shape}")
    return y


def denoise_wga_dapi_joint_raw(img_yxc: np.ndarray) -> np.ndarray:
    """
    Joint WGA+DAPI denoising using model:
      n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw

    """
    img_yxc = np.asarray(img_yxc)
    if img_yxc.ndim != 3 or img_yxc.shape[-1] != 2:
        raise ValueError(f"Expected (Y,X,2), got {img_yxc.shape}")

    model = get_model("2d_wga_dapi", "JOINT_DAPI+WGA_raw")
    return predict_yxc(img_yxc.astype(np.float32, copy=False), model)


def denoise_2d_for_cellpose(img: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper used in the toolkit.

    Accepted input:
      - (Y,X)      -> uses 2d_time single-channel model
      - (Y,X,1)    -> uses 2d_time single-channel model
      - (Y,X,2)    -> uses **JOINT** WGA+DAPI model

    Output:
      - (Y,X,1) or (Y,X,2), float32
    """
    img = np.asarray(img)

    if img.ndim == 2:
        model = get_model("2d_time", "blue")
        y = predict_yxc(img[..., None].astype(np.float32, copy=False), model)
        return y  # (Y,X,1)

    if img.ndim == 3 and img.shape[-1] == 1:
        model = get_model("2d_time", "blue")
        y = predict_yxc(img.astype(np.float32, copy=False), model)
        return y  # (Y,X,1)

    
    if img.ndim == 3 and img.shape[-1] == 2:
        return denoise_wga_dapi_joint_raw(img)

    raise ValueError(f"Unsupported input shape {img.shape}. Expected (Y,X), (Y,X,1) or (Y,X,2).")

