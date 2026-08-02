"""
Intensity-preserving local-threshold filtering for 2D fluorescence images.

The production filter identifies high-intensity candidate pixels from a global
percentile and removes isolated candidates using local spatial support. The
operation is a binary selection operation: retained pixels keep their exact
input value, while rejected pixels are replaced by zero. No normalization,
rescaling, smoothing, or interpolation is performed in this module.

Public functions
----------------
``apply_local_threshold_2d``
    Filter one two-dimensional image plane while preserving its dtype and the
    exact intensity values of retained pixels.
``validate_intensity_preservation``
    Verify that a saved or in-memory filtered plane contains only unchanged
    retained values and zeros at rejected positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.ndimage import maximum_filter, uniform_filter
except Exception:  # pragma: no cover - NumPy fallbacks cover minimal installs
    maximum_filter = None
    uniform_filter = None


@dataclass(frozen=True)
class LocalThresholdParams:
    """Parameters of the production local-threshold filter.

    Parameters
    ----------
    high_percentile:
        Global percentile that defines high-intensity candidate pixels.
    frac3_keep:
        Minimum candidate-pixel fraction in the small neighborhood required to
        retain a candidate directly.
    support3_frac:
        Minimum retained-pixel fraction in the small neighborhood required to
        define a strong local cluster.
    kernel3:
        Width and height of the small neighborhood.
    kernel10:
        Width and height of the broader support neighborhood.
    frac10_keep:
        Minimum candidate-pixel fraction in the broader neighborhood. This
        branch can recover candidates near a strong local cluster.
    """

    high_percentile: float = 99.2
    frac3_keep: float = 0.40
    support3_frac: float = 0.70
    kernel3: int = 3
    kernel10: int = 10
    frac10_keep: float = 0.40

    def validate(self) -> None:
        """Raise ``ValueError`` when a parameter is outside its valid range.

        Raises:
            ValueError: If the supplied inputs or runtime state violate the function's requirements.

        Example:
            >>> instance = LocalThresholdParams(...)
            >>> instance.validate()
        """
        if not 0.0 <= self.high_percentile <= 100.0:
            raise ValueError("high_percentile must be in [0, 100]")
        for name, value in (
            ("frac3_keep", self.frac3_keep),
            ("support3_frac", self.support3_frac),
            ("frac10_keep", self.frac10_keep),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.kernel3 < 1 or self.kernel10 < 1:
            raise ValueError("kernel sizes must be positive integers")

    @property
    def label(self) -> str:
        """Return a stable folder-safe parameter label.

        Returns:
            str: Generated or resolved text value.

        Example:
            >>> instance = LocalThresholdParams(...)
            >>> value = instance.label
        """
        return (
            f"p{self.high_percentile:.1f}_k3f{int(round(self.frac3_keep * 100))}"
            f"_s3{int(round(self.support3_frac * 100))}"
            f"_k10f{int(round(self.frac10_keep * 100))}"
        )

    @property
    def human_title(self) -> str:
        """Return a readable parameter description for reports.

        Returns:
            str: Generated or resolved text value.

        Example:
            >>> instance = LocalThresholdParams(...)
            >>> value = instance.human_title
        """
        return (
            "local-threshold "
            f"(high >= p{self.high_percentile:.1f}, "
            f"{self.kernel3}x{self.kernel3} candidate fraction >= {self.frac3_keep:.2f}, "
            f"local support >= {self.support3_frac:.2f}, "
            f"{self.kernel10}x{self.kernel10} candidate fraction >= {self.frac10_keep:.2f})"
        )


def _mean_filter(mask: np.ndarray, size: int) -> np.ndarray:
    """Return the zero-padded local mean of a Boolean mask.

    Args:
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        size (int): Requested size or size constraint for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _mean_filter(mask=image_array, size=1)
    """
    x = np.asarray(mask, dtype=np.float32)
    if uniform_filter is not None:
        return uniform_filter(x, size=size, mode="constant", cval=0.0)

    pad_before = size // 2
    pad_after = size - 1 - pad_before
    padded = np.pad(
        x,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0.0,
    )
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    window_sum = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (window_sum / float(size * size)).astype(np.float32)


def _maximum_filter(mask: np.ndarray, size: int) -> np.ndarray:
    """Return whether each neighborhood contains at least one true pixel.

    Args:
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        size (int): Requested size or size constraint for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _maximum_filter(mask=image_array, size=1)
    """
    x = np.asarray(mask, dtype=np.uint8)
    if maximum_filter is not None:
        return maximum_filter(x, size=size, mode="constant", cval=0) > 0

    pad_before = size // 2
    pad_after = size - 1 - pad_before
    padded = np.pad(
        x,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0,
    )
    out = np.zeros_like(x, dtype=bool)
    for row_offset in range(size):
        for col_offset in range(size):
            out |= padded[
                row_offset : row_offset + x.shape[0],
                col_offset : col_offset + x.shape[1],
            ].astype(bool)
    return out


def apply_local_threshold_2d(
    image: np.ndarray,
    params: LocalThresholdParams | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Apply local-threshold filtering to one image plane.

    Processing sequence
    -------------------
    1. Calculate the selected global intensity percentile.
    2. Mark pixels at or above that threshold as candidates.
    3. Retain candidates with sufficient candidate support in the small window.
    4. Identify strong small-window clusters.
    5. Recover additional candidates that lie near a strong cluster and also
       satisfy the broader-window candidate-fraction requirement.
    6. Copy original values at retained positions and write zero elsewhere.

    Args:
        image (np.ndarray): Input image array to process.
        params (LocalThresholdParams | None): Value specifying params for the operation. ``None`` selects the function's default behavior.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]: Mapping containing the generated or resolved values.

    Raises:
        TypeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = apply_local_threshold_2d(image=image_array)
    """
    chosen = params or LocalThresholdParams()
    chosen.validate()

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D image plane, received shape={source.shape}")
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(source.dtype, np.bool_):
        raise TypeError(f"Expected a numeric image dtype, received {source.dtype}")
    if not np.all(np.isfinite(source)):
        raise ValueError("Local-threshold input contains NaN or infinite values")

    working = source.astype(np.float64, copy=False)
    threshold = float(np.percentile(working, chosen.high_percentile))
    high_mask = working >= threshold

    frac3_high = _mean_filter(high_mask, size=chosen.kernel3)
    keep3 = high_mask & (frac3_high >= chosen.frac3_keep)

    frac3_keep = _mean_filter(keep3, size=chosen.kernel3)
    strong_cluster = frac3_keep >= chosen.support3_frac
    near_strong_cluster = _maximum_filter(strong_cluster, size=chosen.kernel10)

    frac10_high = _mean_filter(high_mask, size=chosen.kernel10)
    keep10 = high_mask & near_strong_cluster & (frac10_high >= chosen.frac10_keep)
    final_keep = keep3 | keep10

    filtered = np.zeros_like(source)
    filtered[final_keep] = source[final_keep]

    info: dict[str, float | int | str] = {
        "high_threshold_raw": threshold,
        "input_dtype": str(source.dtype),
        "output_dtype": str(filtered.dtype),
        "n_pixels": int(source.size),
        "n_high": int(np.count_nonzero(high_mask)),
        "n_keep3": int(np.count_nonzero(keep3)),
        "n_strong_cluster": int(np.count_nonzero(strong_cluster)),
        "n_keep10": int(np.count_nonzero(keep10)),
        "n_final": int(np.count_nonzero(final_keep)),
        "frac_final": float(np.mean(final_keep)),
    }
    return filtered, final_keep, info


def validate_intensity_preservation(
    original: np.ndarray,
    filtered: np.ndarray,
    keep_mask: np.ndarray,
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Validate that local thresholding changed only rejected pixels to zero.

    Integer data are checked exactly. Floating-point data use ``atol`` with zero
    relative tolerance. The returned dictionary is designed for direct CSV
    export by the validation script.

    Args:
        original (np.ndarray): Array containing original.
        filtered (np.ndarray): Array containing filtered.
        keep_mask (np.ndarray): Array containing keep mask.
        atol (float): Numerical value controlling atol. Defaults to ``0.0``.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = validate_intensity_preservation(
        ...     original=image_array,
        ...     filtered=image_array,
        ...     keep_mask=image_array,
        ... )
    """
    source = np.asarray(original)
    result = np.asarray(filtered)
    mask = np.asarray(keep_mask, dtype=bool)

    shape_match = source.shape == result.shape == mask.shape
    dtype_match = source.dtype == result.dtype
    if not shape_match:
        return {
            "status": "FAIL",
            "shape_match": False,
            "dtype_match": dtype_match,
            "kept_mismatch_count": -1,
            "rejected_nonzero_count": -1,
            "max_abs_kept_error": float("nan"),
        }

    if np.issubdtype(source.dtype, np.integer) or atol == 0.0:
        kept_equal = result[mask] == source[mask]
    else:
        kept_equal = np.isclose(result[mask], source[mask], rtol=0.0, atol=atol)

    kept_mismatch_count = int(np.count_nonzero(~kept_equal))
    rejected_nonzero_count = int(np.count_nonzero(result[~mask] != 0))
    if np.any(mask):
        max_abs_kept_error = float(
            np.max(
                np.abs(
                    result[mask].astype(np.float64)
                    - source[mask].astype(np.float64)
                )
            )
        )
    else:
        max_abs_kept_error = 0.0

    passed = (
        shape_match
        and dtype_match
        and kept_mismatch_count == 0
        and rejected_nonzero_count == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "kept_mismatch_count": kept_mismatch_count,
        "rejected_nonzero_count": rejected_nonzero_count,
        "max_abs_kept_error": max_abs_kept_error,
    }


# Backward-compatible names used by earlier scripts.
LocalHighThresholdSpec = LocalThresholdParams


def apply_local_high_threshold_single(
    raw_img: np.ndarray,
    spec: LocalHighThresholdSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Backward-compatible wrapper for :func:`apply_local_threshold_2d`.

    Args:
        raw_img (np.ndarray): Array containing raw img.
        spec (LocalHighThresholdSpec): Value specifying spec for the operation.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = apply_local_high_threshold_single(raw_img=image_array, spec=...)
    """
    return apply_local_threshold_2d(raw_img, spec)
