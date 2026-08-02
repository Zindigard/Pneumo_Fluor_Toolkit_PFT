"""Provide command-line and programmatic utilities for image utils."""

from __future__ import annotations

import numpy as np

"""Shared image, channel and dtype conversion helpers for 2D pipelines."""

EPS = 1e-12


def normalize01_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    """Normalize an image to [0, 1] using percentile clipping.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        p_lo (float): Numerical value controlling p lo. Defaults to ``1.0``.
        p_hi (float): Numerical value controlling p hi. Defaults to ``99.5``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = normalize01_percentile(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def to_uint8_percentile(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    """Convert an image to uint8 after percentile-based normalization.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        p_lo (float): Numerical value controlling p lo. Defaults to ``1.0``.
        p_hi (float): Numerical value controlling p hi. Defaults to ``99.5``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = to_uint8_percentile(x=image_array)
    """
    return np.clip(255.0 * normalize01_percentile(x, p_lo=p_lo, p_hi=p_hi), 0, 255).astype(np.uint8)


def to_uint8_minmax(x: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 using full min-max normalization.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = to_uint8_minmax(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    out = (x - x_min) / (x_max - x_min)
    return np.clip(255.0 * out, 0, 255).astype(np.uint8)


def dtype_max(arr: np.ndarray) -> float:
    """Return a safe display denominator for integer or float arrays.

    Args:
        arr (np.ndarray): Array containing arr.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = dtype_max(arr=image_array)
    """
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    m = float(np.nanmax(arr)) if arr.size else 1.0
    return m if m > 0 else 1.0


def linear01(img: np.ndarray, denom: float | None = None) -> np.ndarray:
    """Scale an image linearly to [0, 1] with an explicit or inferred denominator.

    Args:
        img (np.ndarray): Array containing img.
        denom (float | None): Numerical value controlling denom. ``None`` selects the function's default behavior.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = linear01(img=image_array)
    """
    img = np.asarray(img, dtype=np.float32)
    d = float(dtype_max(img) if denom is None else denom)
    out = img / d
    return np.clip(out, 0.0, 1.0)


def minmax01(arr: np.ndarray) -> np.ndarray:
    """Map an image to [0, 1] with min-max normalization.

    Args:
        arr (np.ndarray): Array containing arr.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = minmax01(arr=image_array)
    """
    a = np.asarray(arr, dtype=np.float32)
    if not a.size:
        return np.zeros_like(a, dtype=np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def extract_display_plane(x: np.ndarray, axes: str, channel_index: int) -> np.ndarray:
    """Extract one display-ready 2D channel plane from an array and axes string.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        axes (str): Axis specification describing the dimensional order of the image data.
        channel_index (int): Zero-based index selecting channel.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = extract_display_plane(
        ...     x=image_array,
        ...     axes="axes",
        ...     channel_index=1,
        ... )
    """
    if 'c' in axes:
        plane = np.take(x, indices=channel_index, axis=axes.index('c'))
    else:
        plane = x
    if 't' in axes and plane.ndim == 3:
        plane = plane[0]
    return np.asarray(plane, dtype=np.float32)


def rgb_time_norm(arr2d: np.ndarray) -> np.ndarray:
    """Build a blue RGB preview for a single-channel normalized image.

    Args:
        arr2d (np.ndarray): Array containing arr2d.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_time_norm(arr2d=image_array)
    """
    b = normalize01_percentile(arr2d)
    z = np.zeros_like(b)
    return np.dstack([z, z, b])


def rgb_time_raw(arr2d: np.ndarray, denom: float | None = None) -> np.ndarray:
    """Build a blue RGB preview for a single-channel raw display image.

    Args:
        arr2d (np.ndarray): Array containing arr2d.
        denom (float | None): Numerical value controlling denom. ``None`` selects the function's default behavior.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_time_raw(arr2d=image_array)
    """
    b = linear01(arr2d, denom=denom)
    z = np.zeros_like(b)
    return np.dstack([z, z, b])


def rgb_wga_dapi_norm(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Build a blue-green normalized RGB preview for two channels.

    Args:
        blue (np.ndarray): Array containing blue.
        green (np.ndarray): Array containing green.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_wga_dapi_norm(blue=image_array, green=image_array)
    """
    b = normalize01_percentile(blue)
    g = normalize01_percentile(green)
    z = np.zeros_like(b)
    return np.dstack([z, g, b])


def scale_shared_raw(*channels: np.ndarray) -> list[np.ndarray]:
    """Scale multiple channels with a shared denominator for fair raw comparison.

    Args:
        *channels (np.ndarray): Channel indices or identifiers selected for processing.

    Returns:
        list[np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = scale_shared_raw()
    """
    den = max(dtype_max(ch) for ch in channels) if channels else 1.0
    return [linear01(ch, den) for ch in channels]


def rgb_wga_dapi_raw_shared(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Build a blue-green raw RGB preview using one shared scaling denominator.

    Args:
        blue (np.ndarray): Array containing blue.
        green (np.ndarray): Array containing green.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_wga_dapi_raw_shared(blue=image_array, green=image_array)
    """
    b, g = scale_shared_raw(blue, green)
    z = np.zeros_like(b)
    return np.dstack([z, g, b])
