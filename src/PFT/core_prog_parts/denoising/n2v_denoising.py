"""
Load and apply trained two-dimensional Noise2Void models.

Predictions operate on raw-range ``float32`` arrays.  No percentile, min-max,
or display normalization is applied before or after model inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from n2v.models import N2V

from PFT.core_prog_parts.denoising.n2v_workflow import (
    MODEL_SPECS,
    get_model_spec,
    model_key_from_dataset_and_loader_key,
    models_root,
    project_root,
)


MODEL_NAMES = {
    ("2d_time", "blue"): MODEL_SPECS["time"].model_name,
    ("2d_time", "time_ch0"): MODEL_SPECS["time"].model_name,
    ("2d_wga_dapi", "blue_DAPI"): MODEL_SPECS["dapi"].model_name,
    ("2d_wga_dapi", "DAPI_blue"): MODEL_SPECS["dapi"].model_name,
    ("2d_wga_dapi", "green_WGA"): MODEL_SPECS["wga"].model_name,
    ("2d_wga_dapi", "WGA_green"): MODEL_SPECS["wga"].model_name,
    ("2d_wga_dapi", "JOINT_DAPI+WGA_raw"): MODEL_SPECS["joint"].model_name,
    ("2d_wga_dapi", "joint"): MODEL_SPECS["joint"].model_name,
}

_MODEL_CACHE: Dict[Tuple[str, str], N2V] = {}
_MODEL_CACHE_BY_NAME: Dict[str, N2V] = {}


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Return the repository root from an optional starting path."""

    return project_root(start or Path(__file__))


def models_dir() -> Path:
    """Return the directory containing trained N2V model folders."""

    return models_root(_find_repo_root())


def _load_model_by_name(model_name: str) -> N2V:
    """Load one model folder once and cache the N2V object."""

    if model_name in _MODEL_CACHE_BY_NAME:
        return _MODEL_CACHE_BY_NAME[model_name]

    model_base = models_dir()
    model_path = model_base / model_name
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"N2V model directory not found: {model_path}. "
            "Train the corresponding model with scripts/denoising/train_n2v_2d.py."
        )

    model = N2V(config=None, name=model_name, basedir=str(model_base))
    print(f"[N2V] Loaded model: {model_name}")
    print(f"[N2V] Model directory: {model_path}")
    print(f"[N2V] weights_best.h5 present: {(model_path / 'weights_best.h5').exists()}")
    _MODEL_CACHE_BY_NAME[model_name] = model
    return model


def get_model(dataset: str, key: str) -> N2V:
    """Load a model through the legacy ``(dataset, key)`` toolkit interface."""

    cache_key = (str(dataset), str(key))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    model_key = model_key_from_dataset_and_loader_key(dataset, key)
    model = _load_model_by_name(get_model_spec(model_key).model_name)
    _MODEL_CACHE[cache_key] = model
    return model


def load_n2v_model(model_name: str) -> N2V:
    """Load a model by its exact folder name below ``models``."""

    return _load_model_by_name(str(model_name))


def load_n2v_model_by_key(model_key: str) -> N2V:
    """Load a model by the command-line key ``time``, ``dapi``, ``wga``, or ``joint``."""

    return _load_model_by_name(get_model_spec(model_key).model_name)


def predict_yxc(img_yxc: np.ndarray, model: N2V) -> np.ndarray:
    """Predict one raw-range image in ``Y,X,C`` order and return ``float32`` data.

    A batch dimension is intentionally not added because N2V receives one image
    with the explicit axes string ``YXC``.  The result must retain exactly the
    input shape.
    """

    image = np.asarray(img_yxc, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected a YXC image, got shape {image.shape}.")
    try:
        prediction = model.predict(image, axes="YXC")
    except TypeError:
        prediction = model.predict(image)
    output = np.asarray(prediction, dtype=np.float32)
    if output.shape != image.shape:
        raise ValueError(f"Unexpected N2V prediction shape {output.shape}; expected {image.shape}.")
    if not np.all(np.isfinite(output)):
        raise ValueError("N2V prediction contains non-finite values.")
    return output


def denoise_wga_dapi_joint_raw(img_yxc: np.ndarray) -> np.ndarray:
    """Denoise a raw-range ``Y,X,2`` DAPI+WGA image with the joint model."""

    image = np.asarray(img_yxc)
    if image.ndim != 3 or image.shape[-1] != 2:
        raise ValueError(f"Expected shape (Y,X,2), got {image.shape}.")
    return predict_yxc(image, load_n2v_model_by_key("joint"))


def denoise_2d_for_cellpose(img: np.ndarray) -> np.ndarray:
    """Denoise toolkit input before Cellpose without intensity normalization.

    ``Y,X`` and ``Y,X,1`` inputs use the time-lapse single-channel model.
    ``Y,X,2`` inputs use the joint DAPI+WGA model.  The returned array always
    has a final channel axis and ``float32`` dtype.
    """

    image = np.asarray(img)
    if image.ndim == 2:
        return predict_yxc(image[..., None], load_n2v_model_by_key("time"))
    if image.ndim == 3 and image.shape[-1] == 1:
        return predict_yxc(image, load_n2v_model_by_key("time"))
    if image.ndim == 3 and image.shape[-1] == 2:
        return denoise_wga_dapi_joint_raw(image)
    raise ValueError(
        f"Unsupported input shape {image.shape}. Expected (Y,X), (Y,X,1), or (Y,X,2)."
    )
