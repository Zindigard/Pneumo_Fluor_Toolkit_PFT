r"""Dataset-specific single-channel input adapter for StarDist.

The prepared PFT segmentation inputs remain unchanged on disk:

* ``2d_time`` stores one numerical HADA channel.
* ``2d_wga_dapi`` stores two independently normalized DAPI/WGA channels.
* ``3d_mip`` stores three independently normalized HADA/NADA/TADA MIP channels.

The pretrained and fine-tuned ``StarDist2D`` models used by PFT expect one
numerical input channel. Therefore, only inside StarDist processes, multichannel
prepared inputs are converted to one float32 image by a pixelwise maximum:

* DAPI/WGA -> ``max(DAPI, WGA)``
* HADA/NADA/TADA MIP -> ``max(HADA, NADA, TADA)``

The same adapter is installed for training-data collection and all supported
StarDist prediction methods. This keeps screening, fine-tuning, validation, and
production inference consistent.

Examples
--------
Programmatic conversion of a CYX three-channel MIP::

    import numpy as np

    from scripts.segmentation.stardist_input_adapter import (
        merge_stardist_input,
    )

    image_cyx = np.zeros((3, 256, 256), dtype=np.float32)
    merged_yx = merge_stardist_input(image_cyx, dataset="3d_mip")
    assert merged_yx.shape == (256, 256)

Install the monkey patches before StarDist training or inference::

    from scripts.segmentation.stardist_input_adapter import (
        install_stardist_input_adapter,
    )

    install_stardist_input_adapter()
"""

from __future__ import annotations

from functools import wraps
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SCRIPT_FILE = Path(__file__).resolve()
SUPPORTED_DATASETS = frozenset({"2d_time", "2d_wga_dapi", "3d_mip"})


def _project_root() -> Path:
    """Return the PFT repository root.

    Returns
    -------
    pathlib.Path
        Directory containing both ``scripts`` and ``src/PFT``.

    Raises
    ------
    RuntimeError
        If the repository root cannot be determined from this file location.

    Examples
    --------
    >>> root = _project_root()
    >>> (root / "scripts").is_dir()
    True
    """
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (
            candidate / "src" / "PFT"
        ).is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


def _ensure_project_imports() -> Path:
    """Add the repository ``src`` directory to ``sys.path`` when required.

    Returns
    -------
    pathlib.Path
        Resolved PFT repository root.

    Examples
    --------
    >>> root = _ensure_project_imports()
    >>> str(root / "src") in sys.path
    True
    """
    root = _project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _normalize_dataset_name(dataset: str) -> str:
    """Return a validated canonical dataset name.

    Parameters
    ----------
    dataset:
        PFT dataset identifier.

    Returns
    -------
    str
        Lowercase canonical identifier.

    Raises
    ------
    ValueError
        If the dataset identifier is empty.

    Examples
    --------
    >>> _normalize_dataset_name("3D_MIP")
    '3d_mip'
    """
    normalized = str(dataset).strip().lower()
    if not normalized:
        raise ValueError("Dataset name must not be empty.")
    return normalized


def _single_channel_yx(image: np.ndarray, *, dataset: str) -> np.ndarray:
    """Convert a one-channel YX/YXC/CYX image to a float32 YX array.

    Parameters
    ----------
    image:
        Numerical prepared input.
    dataset:
        Dataset name used in diagnostic errors.

    Returns
    -------
    numpy.ndarray
        Two-dimensional float32 image.

    Raises
    ------
    ValueError
        If the input does not contain exactly one numerical channel.

    Examples
    --------
    >>> image = np.zeros((1, 32, 32), dtype=np.float32)
    >>> _single_channel_yx(image, dataset="2d_time").shape
    (32, 32)
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError(
        f"StarDist {dataset} input must contain exactly one numerical "
        f"channel. Received shape {x.shape}."
    )


def _channels_last(
    image: np.ndarray,
    *,
    expected_channels: int,
    dataset: str,
    channel_names: Sequence[str],
) -> np.ndarray:
    """Return a multichannel image in YXC order.

    Parameters
    ----------
    image:
        Prepared input in YXC or CYX order.
    expected_channels:
        Required number of numerical channels.
    dataset:
        Dataset name used in diagnostic errors.
    channel_names:
        Human-readable channel names used in diagnostic errors.

    Returns
    -------
    numpy.ndarray
        Float32 image in YXC order.

    Raises
    ------
    ValueError
        If the channel count or array layout is unsupported.

    Examples
    --------
    >>> image = np.zeros((3, 32, 32), dtype=np.float32)
    >>> result = _channels_last(
    ...     image,
    ...     expected_channels=3,
    ...     dataset="3d_mip",
    ...     channel_names=("HADA", "NADA", "TADA"),
    ... )
    >>> result.shape
    (32, 32, 3)
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] == expected_channels:
        return x
    if x.ndim == 3 and x.shape[0] == expected_channels:
        return np.moveaxis(x, 0, -1)

    names = ", ".join(channel_names)
    raise ValueError(
        f"StarDist {dataset} input must contain {expected_channels} numerical "
        f"channels ({names}) in YXC or CYX order. Received shape {x.shape}."
    )


def _validate_merged_yx(image: np.ndarray, *, dataset: str) -> np.ndarray:
    """Validate and return the final float32 YX image supplied to StarDist.

    Parameters
    ----------
    image:
        Candidate merged image.
    dataset:
        Dataset name used in diagnostic errors.

    Returns
    -------
    numpy.ndarray
        Finite float32 image in YX order, clipped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If the merged result is not a finite two-dimensional image.

    Examples
    --------
    >>> merged = np.ones((16, 16), dtype=np.float32)
    >>> _validate_merged_yx(merged, dataset="3d_mip").shape
    (16, 16)
    """
    merged = np.asarray(image, dtype=np.float32)
    if merged.ndim != 2:
        raise ValueError(
            f"The merged StarDist {dataset} input must have YX layout. "
            f"Received shape {merged.shape}."
        )
    if not np.all(np.isfinite(merged)):
        raise ValueError(
            f"The merged StarDist {dataset} input contains NaN or infinite "
            "values."
        )
    return np.clip(merged, 0.0, 1.0).astype(np.float32, copy=False)


def merge_stardist_input(image: np.ndarray, dataset: str) -> np.ndarray:
    """Return the numerical single-channel image supplied to StarDist.

    Parameters
    ----------
    image:
        Prepared float32 PFT segmentation input. Supported layouts are YX,
        YXC, CYX, YX1, and 1YX.
    dataset:
        PFT dataset name: ``2d_time``, ``2d_wga_dapi``, or ``3d_mip``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional float32 StarDist input for supported datasets. For an
        unknown dataset, the original float32 array is returned unchanged.

    Raises
    ------
    ValueError
        If a supported dataset has an invalid channel count or layout.

    Examples
    --------
    One-channel time-lapse input::

        result = merge_stardist_input(image, dataset="2d_time")

    Three-channel MIP input::

        result = merge_stardist_input(image, dataset="3d_mip")
    """
    dataset_name = _normalize_dataset_name(dataset)
    x = np.asarray(image, dtype=np.float32)

    if dataset_name == "2d_time":
        return _validate_merged_yx(
            _single_channel_yx(x, dataset=dataset_name),
            dataset=dataset_name,
        )

    if dataset_name == "2d_wga_dapi":
        channels = _channels_last(
            x,
            expected_channels=2,
            dataset=dataset_name,
            channel_names=("DAPI", "WGA"),
        )
        # The channels were normalized independently before the shared U-Net
        # mask was applied. Maximum projection preserves evidence from either
        # the nucleoid or the cell-wall signal without additional scaling.
        merged = np.max(channels, axis=-1)
        return _validate_merged_yx(merged, dataset=dataset_name)

    if dataset_name == "3d_mip":
        channels = _channels_last(
            x,
            expected_channels=3,
            dataset=dataset_name,
            channel_names=("HADA", "NADA", "TADA"),
        )
        # HADA, NADA, and TADA MIPs were normalized independently. The
        # pixelwise maximum retains fluorescence evidence from all sequential
        # cell-wall labels while producing the one-channel input expected by
        # StarDist2D.
        merged = np.max(channels, axis=-1)
        return _validate_merged_yx(merged, dataset=dataset_name)

    # Preserve historical behavior for future/unknown datasets. The caller may
    # apply a separate dataset-specific policy instead of silently merging it.
    return x


def _dataset_from_prediction_image(image: np.ndarray) -> str | None:
    """Infer the supported PFT dataset policy from an image shape.

    Parameters
    ----------
    image:
        Prepared prediction input.

    Returns
    -------
    str or None
        Inferred dataset policy, or ``None`` when the shape is unsupported.

    Notes
    -----
    This shape-based inference is used only inside the StarDist monkey patch.
    The explicit dataset argument remains the source of truth during training
    collection.

    Examples
    --------
    >>> _dataset_from_prediction_image(np.zeros((3, 64, 64)))
    '3d_mip'
    """
    x = np.asarray(image)
    if x.ndim == 2:
        return "2d_time"
    if x.ndim == 3 and (x.shape[-1] == 3 or x.shape[0] == 3):
        return "3d_mip"
    if x.ndim == 3 and (x.shape[-1] == 2 or x.shape[0] == 2):
        return "2d_wga_dapi"
    if x.ndim == 3 and (x.shape[-1] == 1 or x.shape[0] == 1):
        return "2d_time"
    return None


def _dataset_from_training_call(
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> str | None:
    """Resolve a dataset identifier passed to ``collect_training_data``.

    Parameters
    ----------
    args:
        Positional arguments supplied to the original collector.
    kwargs:
        Keyword arguments supplied to the original collector.

    Returns
    -------
    str or None
        Canonical supported dataset identifier, when found.

    Examples
    --------
    >>> _dataset_from_training_call(("cellpose", "3d_mip"), {})
    '3d_mip'
    """
    explicit = kwargs.get("dataset")
    if explicit is not None:
        candidate = str(explicit).strip().lower()
        return candidate if candidate in SUPPORTED_DATASETS else None

    # Preserve compatibility with the current collector signature, where the
    # dataset is normally the second positional argument, while remaining
    # robust if the argument order changes in a later refactor.
    if len(args) >= 2:
        candidate = str(args[1]).strip().lower()
        if candidate in SUPPORTED_DATASETS:
            return candidate

    for value in args:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in SUPPORTED_DATASETS:
                return candidate
    return None


def _patch_training_collection() -> None:
    """Patch PFT training collection with the StarDist channel policy.

    The patch is idempotent. Images are converted only after the regular PFT
    collector has paired and validated the training inputs and masks.

    Examples
    --------
    >>> _patch_training_collection()  # doctest: +SKIP
    """
    from PFT.core_prog_parts.segmentation import (
        instance_segmentation_core as core,
    )

    if getattr(core, "_pft_stardist_merge_collect_installed", False):
        return

    original: Callable[
        ..., tuple[list[np.ndarray], list[np.ndarray], list[str]]
    ] = core.collect_training_data

    @wraps(original)
    def collect_training_data_stardist(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        """Collect training pairs and adapt supported images for StarDist.

        Parameters
        ----------
        *args, **kwargs:
            Arguments forwarded unchanged to the original PFT collector.

        Returns
        -------
        tuple
            Adapted images, original integer masks, and sample names.
        """
        images, masks, names = original(*args, **kwargs)
        dataset = _dataset_from_training_call(args, kwargs)
        if dataset is not None:
            images = [
                merge_stardist_input(image, dataset)
                for image in images
            ]
        return images, masks, names

    core.collect_training_data = collect_training_data_stardist
    core._pft_stardist_merge_collect_installed = True


def _patch_stardist_prediction_methods() -> None:
    """Patch StarDist2D prediction methods with automatic PFT input adaptation.

    The patch covers standard prediction, tiled/large-image prediction, and the
    lower-level probability/distance prediction method used during threshold
    optimization. It is safe to call repeatedly.

    Examples
    --------
    >>> _patch_stardist_prediction_methods()  # doctest: +SKIP
    """
    from stardist.models import StarDist2D

    if getattr(
        StarDist2D,
        "_pft_stardist_merge_prediction_installed",
        False,
    ):
        return

    def wrap(method_name: str) -> None:
        """Install one idempotent wrapper on a StarDist2D method.

        Parameters
        ----------
        method_name:
            Name of the StarDist2D method to wrap.
        """
        original = getattr(StarDist2D, method_name, None)
        if original is None or getattr(
            original,
            "_pft_stardist_merge_wrapper",
            False,
        ):
            return

        @wraps(original)
        def wrapped(
            self: Any,
            image: np.ndarray,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Adapt a supported PFT image and call the original method."""
            dataset = _dataset_from_prediction_image(image)
            adapted = (
                merge_stardist_input(image, dataset)
                if dataset is not None
                else np.asarray(image, dtype=np.float32)
            )

            expected_channels = int(
                getattr(getattr(self, "config", None), "n_channel_in", 1)
            )
            if expected_channels == 1 and adapted.ndim != 2:
                raise ValueError(
                    "PFT StarDist input adaptation did not produce a YX "
                    f"single-channel image. Received shape {adapted.shape}."
                )

            return original(self, adapted, *args, **kwargs)

        wrapped._pft_stardist_merge_wrapper = True
        setattr(StarDist2D, method_name, wrapped)

    # ``predict_instances`` is used by standard PFT inference and validation.
    # The other methods support large-image prediction and threshold tuning.
    for name in ("predict_instances", "predict_instances_big", "predict"):
        wrap(name)

    StarDist2D._pft_stardist_merge_prediction_installed = True


def install_stardist_input_adapter() -> None:
    """Install all StarDist-specific PFT input adapters.

    This function should be called once near the beginning of each StarDist
    terminal script. Repeated calls are harmless because both patches are
    idempotent.

    Examples
    --------
    >>> install_stardist_input_adapter()  # doctest: +SKIP
    """
    _ensure_project_imports()
    _patch_training_collection()
    _patch_stardist_prediction_methods()


__all__ = [
    "install_stardist_input_adapter",
    "merge_stardist_input",
]
