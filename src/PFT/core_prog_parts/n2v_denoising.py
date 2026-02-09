from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from n2v.models import N2V


#  Model
MODEL_NAMES = {
    ("2d_time", "blue"): "n2v_2d_time_blue_p64x64",
    ("2d_wga_dapi", "blue_DAPI"): "n2v_2d_wga_dapi_blue_DAPI_p64x64",
    ("2d_wga_dapi", "green_WGA"): "n2v_2d_wga_dapi_green_WGA_p64x64",
}



_MODEL_CACHE: Dict[Tuple[str, str], N2V] = {}


def _project_root_from_this_file() -> Path:
    """
    File location:
     
    """
    return Path(__file__).resolve().parents[3]


def _models_dir() -> Path:
    return _project_root_from_this_file() / "models"


def _get_model(dataset: str, key: str) -> N2V:
    cache_key = (dataset, key)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_dir = _models_dir()
    model_name = MODEL_NAMES[(dataset, key)]

    # Load existing model from disk
    model = N2V(config=None, name=model_name, basedir=str(model_dir))
    _MODEL_CACHE[cache_key] = model
    return model


def _predict_single_channel(img_yx: np.ndarray, model: N2V) -> np.ndarray:
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

    y = model.predict(x, axes="YXC")  # returns (Y,X,1)
    y = np.asarray(y)
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    elif y.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected prediction shape {y.shape} from N2V model.")
    return y


def denoise_2d_for_cellpose(img: np.ndarray) -> np.ndarray:
    """
    Denoise 2D datasets and  return YXC.

    Accepted input:
      - (Y,X)
      - (Y,X,1)
      - (Y,X,2)  with channel order:

    Output:
      - (Y,X,1) for single-channel input
      - (Y,X,2) for two-channel input
    """
    img = np.asarray(img)

    # (Y,X) -> 2d_time single channel
    if img.ndim == 2:
        dataset = "2d_time"
        model = _get_model(dataset, "blue")
        den = _predict_single_channel(img, model)     # (Y,X)
        return den[..., np.newaxis]                   # (Y,X,1)

    # (Y,X,C)
    if img.ndim == 3:
        # (Y,X,1)
        if img.shape[-1] == 1:
            dataset = "2d_time"
            model = _get_model(dataset, "blue")
            den = _predict_single_channel(img[..., 0], model)
            return den[..., np.newaxis]               # (Y,X,1)

        # (Y,X,2) -> 2d_wga_dapi
        if img.shape[-1] == 2:
            dataset = "2d_wga_dapi"
            model_dapi = _get_model(dataset, "blue_DAPI")
            model_wga = _get_model(dataset, "green_WGA")

            den0 = _predict_single_channel(img[..., 0], model_dapi)  # DAPI
            den1 = _predict_single_channel(img[..., 1], model_wga)   # WGA
            return np.stack([den0, den1], axis=-1)                   # (Y,X,2)

    raise ValueError(
        f"Unsupported input shape {img.shape}. Expected (Y,X), (Y,X,1) or (Y,X,2)."
    )
