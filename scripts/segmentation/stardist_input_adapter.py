"""Dataset-specific single-channel input adapter for StarDist.

The prepared PFT inputs remain unchanged on disk:

* ``2d_time`` stays a single HADA channel.
* ``2d_wga_dapi`` remains a two-channel numerical array for other models.

Only inside StarDist processes, the independently normalized DAPI and WGA
channels are merged into one float32 image using a pixelwise maximum. This
keeps the input compatible with the one-channel ``2D_versatile_fluo`` model
and allows its pretrained weights to be transferred correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


def _ensure_project_imports() -> Path:
    root = _project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def merge_stardist_input(image: np.ndarray, dataset: str) -> np.ndarray:
    """Return the numerical image supplied to StarDist.

    Parameters
    ----------
    image:
        Prepared float32 PFT segmentation input. Supported layouts are YX,
        YXC, CYX, YX1, or 1YX.
    dataset:
        PFT dataset name.

    Returns
    -------
    numpy.ndarray
        Single-channel YX float32 image in the original [0, 1] range.
    """

    x = np.asarray(image, dtype=np.float32)

    if dataset == "2d_time":
        if x.ndim == 2:
            return x
        if x.ndim == 3 and x.shape[-1] == 1:
            return x[..., 0]
        if x.ndim == 3 and x.shape[0] == 1:
            return x[0]
        raise ValueError(
            "StarDist 2d_time input must contain one numerical channel. "
            f"Received shape {x.shape}."
        )

    if dataset == "2d_wga_dapi":
        if x.ndim == 3 and x.shape[-1] == 2:
            dapi = x[..., 0]
            wga = x[..., 1]
        elif x.ndim == 3 and x.shape[0] == 2:
            dapi = x[0]
            wga = x[1]
        else:
            raise ValueError(
                "StarDist 2d_wga_dapi input must contain DAPI and WGA as two "
                f"numerical channels. Received shape {x.shape}."
            )

        # Both source channels were already normalized independently to [0, 1]
        # before the shared U-Net mask was applied. A maximum merge preserves
        # cell-wall and nucleoid evidence without another normalization step.
        merged = np.maximum(dapi, wga)
        return np.clip(merged, 0.0, 1.0).astype(np.float32, copy=False)

    # Do not silently alter other datasets.
    return x


def _dataset_from_prediction_image(image: np.ndarray) -> str | None:
    x = np.asarray(image)
    if x.ndim == 2:
        return "2d_time"
    if x.ndim == 3 and (x.shape[-1] == 2 or x.shape[0] == 2):
        return "2d_wga_dapi"
    if x.ndim == 3 and (x.shape[-1] == 1 or x.shape[0] == 1):
        return "2d_time"
    return None


def _patch_training_collection() -> None:
    from PFT.core_prog_parts.segmentation import instance_segmentation_core as core

    if getattr(core, "_pft_stardist_merge_collect_installed", False):
        return

    original: Callable[..., tuple[list[np.ndarray], list[np.ndarray], list[str]]] = (
        core.collect_training_data
    )

    def collect_training_data_stardist(*args: Any, **kwargs: Any):
        images, masks, names = original(*args, **kwargs)

        dataset = kwargs.get("dataset")
        if dataset is None and len(args) >= 2:
            dataset = args[1]
        dataset = str(dataset) if dataset is not None else ""

        if dataset in {"2d_time", "2d_wga_dapi"}:
            images = [merge_stardist_input(image, dataset) for image in images]

        return images, masks, names

    core.collect_training_data = collect_training_data_stardist
    core._pft_stardist_merge_collect_installed = True


def _patch_stardist_prediction_methods() -> None:
    from stardist.models import StarDist2D

    if getattr(StarDist2D, "_pft_stardist_merge_prediction_installed", False):
        return

    def wrap(method_name: str) -> None:
        original = getattr(StarDist2D, method_name, None)
        if original is None or getattr(original, "_pft_stardist_merge_wrapper", False):
            return

        def wrapped(self: Any, image: np.ndarray, *args: Any, **kwargs: Any):
            dataset = _dataset_from_prediction_image(image)
            adapted = merge_stardist_input(image, dataset) if dataset else image
            return original(self, adapted, *args, **kwargs)

        wrapped.__name__ = getattr(original, "__name__", method_name)
        wrapped.__doc__ = getattr(original, "__doc__", None)
        wrapped._pft_stardist_merge_wrapper = True
        setattr(StarDist2D, method_name, wrapped)

    # ``predict_instances`` is used by standard PFT inference and validation.
    # The other methods are patched for threshold optimization and tiled runs.
    for name in ("predict_instances", "predict_instances_big", "predict"):
        wrap(name)

    StarDist2D._pft_stardist_merge_prediction_installed = True


def install_stardist_input_adapter() -> None:
    """Install StarDist-only training and prediction adapters."""

    _ensure_project_imports()
    _patch_training_collection()
    _patch_stardist_prediction_methods()


def policy_description(dataset: str | None) -> str:
    if dataset == "2d_wga_dapi":
        return "single-channel max(DAPI, WGA), no second normalization"
    if dataset == "2d_time":
        return "single HADA channel, unchanged"
    return "automatic StarDist single-channel policy"
