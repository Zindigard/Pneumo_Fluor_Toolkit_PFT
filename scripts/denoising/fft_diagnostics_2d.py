r"""
Structured-noise and FFT diagnostics for two-dimensional fluorescence data.

Parameters
-----------------
``neighbour_correlation``
    Mean of horizontal and vertical adjacent-pixel Pearson correlations.
``fano_factor``
    Full-plane pixel variance divided by full-plane mean intensity.
``fft_peak_score``
    Maximum non-DC Fourier power divided by median non-DC Fourier power.

The three quantities are calculated identically for raw and local-thresholded
planes, allowing direct before/after CSV summaries.

Examples
--------
Programmatic use:

    from scripts.denoising.fft_diagnostics_2d import neighbour_correlation

    result = neighbour_correlation(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class FFTDiagnosticConfig:
    """Configuration for additional frequency-domain measurements.

    Parameters
    ----------
    dc_radius_px:
        Circular radius around the Fourier origin excluded from the extended FFT
        peak and directionality measurements.
    low_frequency_limit:
        Upper normalized radius of the low-frequency band.
    mid_frequency_limit:
        Upper normalized radius of the mid-frequency band.
    angular_axis_half_width_deg:
        Half-width used for horizontal and vertical Fourier-axis power.
    peak_ratio_threshold:
        Peak-to-median threshold used only by the screening flag.
    directionality_threshold:
        Anisotropy threshold used only by the screening flag.
    """

    dc_radius_px: int = 8
    low_frequency_limit: float = 0.10
    mid_frequency_limit: float = 0.35
    angular_axis_half_width_deg: float = 5.0
    peak_ratio_threshold: float = 8.0
    directionality_threshold: float = 0.15

    def validate(self) -> None:
        """Raise ``ValueError`` when a diagnostic parameter is invalid.

        Raises:
            ValueError: If the supplied inputs or runtime state violate the function's requirements.

        Example:
            >>> instance = FFTDiagnosticConfig(...)
            >>> instance.validate()
        """
        if self.dc_radius_px < 0:
            raise ValueError("dc_radius_px must be >= 0")
        if not 0.0 < self.low_frequency_limit < self.mid_frequency_limit < 1.0:
            raise ValueError(
                "Frequency limits must satisfy 0 < low_frequency_limit < "
                "mid_frequency_limit < 1"
            )
        if not 0.0 < self.angular_axis_half_width_deg <= 45.0:
            raise ValueError("angular_axis_half_width_deg must be in (0, 45]")
        if self.peak_ratio_threshold <= 0.0:
            raise ValueError("peak_ratio_threshold must be > 0")
        if not 0.0 <= self.directionality_threshold <= 1.0:
            raise ValueError("directionality_threshold must be in [0, 1]")


def _as_finite_2d(image: np.ndarray) -> np.ndarray:
    """Return a finite ``float64`` 2D array for quantitative calculations.

    Args:
        image (np.ndarray): Input image array to process.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _as_finite_2d(image=image_array)
    """
    x = np.asarray(image)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D image plane, received shape={x.shape}")
    x = x.astype(np.float64, copy=False)
    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        fill = float(np.median(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    return x


def _pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation for equally sized arrays.

    Args:
        first (np.ndarray): Array containing first.
        second (np.ndarray): Array containing second.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = _pearson_correlation(first=image_array, second=image_array)
    """
    first_flat = np.asarray(first, dtype=np.float64).ravel()
    second_flat = np.asarray(second, dtype=np.float64).ravel()
    finite = np.isfinite(first_flat) & np.isfinite(second_flat)
    first_flat = first_flat[finite]
    second_flat = second_flat[finite]
    if first_flat.size < 2:
        return 0.0

    first_flat = first_flat - float(np.mean(first_flat))
    second_flat = second_flat - float(np.mean(second_flat))
    denominator = float(
        np.sqrt(np.sum(first_flat * first_flat) * np.sum(second_flat * second_flat))
    )
    if denominator <= EPS:
        return 0.0
    return float(np.sum(first_flat * second_flat) / denominator)


def neighbour_correlation(image: np.ndarray) -> float:
    """Return mean horizontal/vertical adjacent-pixel Pearson correlation.

    Args:
        image (np.ndarray): Input image array to process.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = neighbour_correlation(image=image_array)
    """
    x = _as_finite_2d(image)
    values: list[float] = []
    if x.shape[1] >= 2:
        values.append(_pearson_correlation(x[:, :-1], x[:, 1:]))
    if x.shape[0] >= 2:
        values.append(_pearson_correlation(x[:-1, :], x[1:, :]))
    return float(np.mean(values)) if values else 0.0


def row_column_adjacent_correlations(image: np.ndarray) -> tuple[float, float]:
    """Return mean correlations between adjacent rows and adjacent columns.

    Args:
        image (np.ndarray): Input image array to process.

    Returns:
        tuple[float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = row_column_adjacent_correlations(image=image_array)
    """
    x = _as_finite_2d(image)
    row_values = [
        _pearson_correlation(x[index, :], x[index + 1, :])
        for index in range(max(0, x.shape[0] - 1))
    ]
    column_values = [
        _pearson_correlation(x[:, index], x[:, index + 1])
        for index in range(max(0, x.shape[1] - 1))
    ]
    row_mean = float(np.mean(row_values)) if row_values else 0.0
    column_mean = float(np.mean(column_values)) if column_values else 0.0
    return row_mean, column_mean


def fano_factor(image: np.ndarray) -> float:
    """Return full-plane variance divided by full-plane mean intensity.

    Args:
        image (np.ndarray): Input image array to process.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = fano_factor(image=image_array)
    """
    x = _as_finite_2d(image)
    mean = float(np.mean(x))
    variance = float(np.var(x))
    if abs(mean) <= EPS:
        return 0.0
    return float(variance / mean)


def fft_peak_score(image: np.ndarray, dc_halfwidth_px: int | None = None) -> float:
    """Return maximum non-DC Fourier power divided by median non-DC power.

    When ``dc_halfwidth_px`` is omitted, the excluded central square follows the
    thesis analysis implementation: ``max(2, min(height, width) // 32)``.

    Args:
        image (np.ndarray): Input image array to process.
        dc_halfwidth_px (int | None): Numerical value controlling dc halfwidth px. ``None`` selects the function's default behavior.

    Returns:
        float: Computed numerical result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = fft_peak_score(image=image_array)
    """
    x = _as_finite_2d(image)
    centered = x - float(np.mean(x))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2

    height, width = power.shape
    center_y, center_x = height // 2, width // 2
    halfwidth = (
        max(2, min(height, width) // 32)
        if dc_halfwidth_px is None
        else int(dc_halfwidth_px)
    )
    if halfwidth < 0:
        raise ValueError("dc_halfwidth_px must be >= 0")

    masked = power.astype(np.float64, copy=True)
    masked[
        max(0, center_y - halfwidth) : min(height, center_y + halfwidth + 1),
        max(0, center_x - halfwidth) : min(width, center_x + halfwidth + 1),
    ] = np.nan
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite) / (np.median(finite) + EPS))


def compute_thesis_structured_noise_metrics(
    image: np.ndarray,
) -> dict[str, float]:
    """Calculate all structured-noise parameters required by the thesis table.

    Args:
        image (np.ndarray): Input image array to process.

    Returns:
        dict[str, float]: Mapping containing the generated or resolved values.

    Example:
        >>> result = compute_thesis_structured_noise_metrics(image=image_array)
    """
    row_corr, column_corr = row_column_adjacent_correlations(image)
    return {
        "neighbour_correlation": neighbour_correlation(image),
        "row_adjacent_correlation": row_corr,
        "column_adjacent_correlation": column_corr,
        "fano_factor": fano_factor(image),
        "fft_peak_score": fft_peak_score(image),
    }


def _frequency_grids(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Return shifted frequency, radius, and angle grids for an image shape.

    Args:
        shape (tuple[int, int]): Target or observed array shape.

    Returns:
        tuple[np.ndarray, ...]: Collection containing the generated or selected values.

    Example:
        >>> result = _frequency_grids(shape=1)
    """
    height, width = shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    u, v = np.meshgrid(fx, fy)
    radius_cyc_px = np.sqrt(u * u + v * v)
    max_radius = float(np.max(radius_cyc_px))
    radius_norm = radius_cyc_px / max(max_radius, EPS)
    angle_deg = np.degrees(np.arctan2(v, u))
    return u, v, radius_cyc_px, radius_norm, angle_deg


def _dc_mask(shape: tuple[int, int], radius_px: int) -> np.ndarray:
    """Return a Boolean mask excluding a circular DC neighborhood.

    Args:
        shape (tuple[int, int]): Target or observed array shape.
        radius_px (int): Numerical value controlling radius px.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _dc_mask(shape=1, radius_px=1)
    """
    height, width = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    cy = float(height // 2)
    cx = float(width // 2)
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > float(radius_px * radius_px)


def _normalize_orientation(angle_deg: float) -> float:
    """Normalize an undirected orientation to ``[-90, 90)``.

    Args:
        angle_deg (float): Numerical value controlling angle deg.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = _normalize_orientation(angle_deg=0.5)
    """
    return float((float(angle_deg) + 90.0) % 180.0 - 90.0)


def _directionality(power: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    """Return Fourier-power anisotropy and principal frequency direction.

    Args:
        power (np.ndarray): Array containing power.
        valid_mask (np.ndarray): Array containing valid mask.

    Returns:
        tuple[float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = _directionality(power=image_array, valid_mask=image_array)
    """
    weighted = np.where(valid_mask, power, 0.0)
    total = float(np.sum(weighted))
    if total <= EPS:
        return 0.0, 0.0

    height, width = power.shape
    yy, xx = np.indices(power.shape, dtype=np.float64)
    cx = float(width // 2)
    cy = float(height // 2)
    u = xx - cx
    v = yy - cy

    m_xx = float(np.sum(weighted * u * u) / total)
    m_yy = float(np.sum(weighted * v * v) / total)
    m_xy = float(np.sum(weighted * u * v) / total)
    trace = m_xx + m_yy
    root = float(np.sqrt(max((m_xx - m_yy) ** 2 + 4.0 * m_xy * m_xy, 0.0)))
    lambda_1 = 0.5 * (trace + root)
    lambda_2 = 0.5 * (trace - root)
    score = float(
        np.clip((lambda_1 - lambda_2) / (lambda_1 + lambda_2 + EPS), 0.0, 1.0)
    )
    angle = 0.5 * np.degrees(np.arctan2(2.0 * m_xy, m_xx - m_yy))
    return score, _normalize_orientation(float(angle))


def _band_fraction(power: np.ndarray, mask: np.ndarray, valid_mask: np.ndarray) -> float:
    """Return the fraction of non-DC power in one frequency band.

    Args:
        power (np.ndarray): Array containing power.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        valid_mask (np.ndarray): Array containing valid mask.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = _band_fraction(
        ...     power=image_array,
        ...     mask=image_array,
        ...     valid_mask=image_array,
        ... )
    """
    denominator = float(np.sum(power[valid_mask]))
    if denominator <= EPS:
        return 0.0
    return float(np.sum(power[mask & valid_mask]) / denominator)


def compute_fft_diagnostics(
    image: np.ndarray,
    config: FFTDiagnosticConfig | None = None,
) -> dict[str, float | int | bool]:
    """Calculate thesis parameters and extended FFT measurements for one plane.

    Args:
        image (np.ndarray): Input image array to process.
        config (FFTDiagnosticConfig | None): Configuration object containing the parameters required by the workflow. ``None`` selects the function's default behavior.

    Returns:
        dict[str, float | int | bool]: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = compute_fft_diagnostics(image=image_array)
    """
    cfg = config or FFTDiagnosticConfig()
    cfg.validate()
    x = _as_finite_2d(image)
    height, width = x.shape

    centered = x - float(np.mean(x))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    valid = _dc_mask(x.shape, cfg.dc_radius_px)
    valid_power = power[valid]

    peak_to_median = 0.0
    peak_to_p95 = 0.0
    peak_frequency = 0.0
    peak_period = 0.0
    peak_angle = 0.0
    peak_offset_x_px = 0.0
    peak_offset_y_px = 0.0
    peak_radius_px = 0.0
    peak_radius_normalized = 0.0

    u, v, radius_cyc_px, radius_norm, angle_deg = _frequency_grids(x.shape)
    if valid_power.size and float(np.sum(valid_power)) > EPS:
        baseline_median = float(np.median(valid_power)) + EPS
        baseline_p95 = float(np.percentile(valid_power, 95.0)) + EPS
        half_plane = (v < 0.0) | ((np.abs(v) <= EPS) & (u >= 0.0))
        candidate_power = np.where(valid & half_plane, power, -np.inf)
        peak_index = np.unravel_index(int(np.argmax(candidate_power)), power.shape)
        peak_value = float(power[peak_index])

        peak_to_median = peak_value / baseline_median
        peak_to_p95 = peak_value / baseline_p95
        peak_frequency = float(radius_cyc_px[peak_index])
        peak_period = float(1.0 / peak_frequency) if peak_frequency > EPS else 0.0
        peak_angle = _normalize_orientation(float(angle_deg[peak_index]))
        peak_offset_y_px = float(peak_index[0] - height // 2)
        peak_offset_x_px = float(peak_index[1] - width // 2)
        peak_radius_px = float(np.hypot(peak_offset_x_px, peak_offset_y_px))
        max_radius_px = float(np.hypot(width // 2, height // 2))
        peak_radius_normalized = peak_radius_px / max(max_radius_px, EPS)

    directionality_score, directionality_angle = _directionality(power, valid)
    low_mask = radius_norm <= cfg.low_frequency_limit
    mid_mask = (radius_norm > cfg.low_frequency_limit) & (
        radius_norm <= cfg.mid_frequency_limit
    )
    high_mask = radius_norm > cfg.mid_frequency_limit

    horizontal_distance = np.abs(((angle_deg + 90.0) % 180.0) - 90.0)
    vertical_distance = np.abs(
        np.abs(((angle_deg + 90.0) % 180.0) - 90.0) - 90.0
    )
    horizontal_axis_mask = horizontal_distance <= cfg.angular_axis_half_width_deg
    vertical_axis_mask = vertical_distance <= cfg.angular_axis_half_width_deg

    p1, p50, p99, p992, p995 = np.percentile(x, [1.0, 50.0, 99.0, 99.2, 99.5])
    mean = float(np.mean(x))
    variance = float(np.var(x))
    std = float(np.sqrt(variance))
    structured_flag = bool(
        peak_to_median >= cfg.peak_ratio_threshold
        and directionality_score >= cfg.directionality_threshold
    )

    thesis_metrics = compute_thesis_structured_noise_metrics(x)
    return {
        "height": int(height),
        "width": int(width),
        "intensity_mean": mean,
        "intensity_std": std,
        "intensity_variance": variance,
        "intensity_p01": float(p1),
        "intensity_p50": float(p50),
        "intensity_p99": float(p99),
        "intensity_p99_2": float(p992),
        "intensity_p99_5": float(p995),
        **thesis_metrics,
        "fft_peak_to_median": float(peak_to_median),
        "fft_peak_to_p95": float(peak_to_p95),
        "fft_peak_frequency_cyc_per_px": float(peak_frequency),
        "fft_peak_period_px": float(peak_period),
        "fft_peak_offset_x_px": float(peak_offset_x_px),
        "fft_peak_offset_y_px": float(peak_offset_y_px),
        "fft_peak_radius_px": float(peak_radius_px),
        "fft_peak_radius_normalized": float(peak_radius_normalized),
        "fft_peak_angle_deg": float(peak_angle),
        "stripe_orientation_deg": _normalize_orientation(peak_angle + 90.0),
        "fft_directionality_score": float(directionality_score),
        "fft_directionality_angle_deg": float(directionality_angle),
        "low_frequency_power_fraction": _band_fraction(power, low_mask, valid),
        "mid_frequency_power_fraction": _band_fraction(power, mid_mask, valid),
        "high_frequency_power_fraction": _band_fraction(power, high_mask, valid),
        "horizontal_axis_power_fraction": _band_fraction(
            power, horizontal_axis_mask, valid
        ),
        "vertical_axis_power_fraction": _band_fraction(
            power, vertical_axis_mask, valid
        ),
        "structured_noise_flag": structured_flag,
        "dc_radius_px": int(cfg.dc_radius_px),
        "peak_ratio_threshold": float(cfg.peak_ratio_threshold),
        "directionality_threshold": float(cfg.directionality_threshold),
    }


def fft_log_magnitude_uint8(image: np.ndarray) -> np.ndarray:
    """Return a display-only percentile-scaled FFT log-magnitude preview.

    Args:
        image (np.ndarray): Input image array to process.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = fft_log_magnitude_uint8(image=image_array)
    """
    x = _as_finite_2d(image)
    spectrum = np.fft.fftshift(np.fft.fft2(x - float(np.mean(x))))
    view = np.log1p(np.abs(spectrum))
    lo, hi = np.percentile(view, [1.0, 99.8])
    if hi <= lo:
        return np.zeros(view.shape, dtype=np.uint8)
    scaled = np.clip((view - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(255.0 * scaled).astype(np.uint8)


def iter_2d_planes(
    array: np.ndarray,
    axes: str,
    *,
    max_frames: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(channel_index, frame_index, plane)`` from an image array.

    Args:
        array (np.ndarray): Array containing array.
        axes (str): Axis specification describing the dimensional order of the image data.
        max_frames (int | None): Maximum permitted value of frames. ``None`` selects the function's default behavior.

    Returns:
        Iterator[tuple[int, int, np.ndarray]]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = iter_2d_planes(array=image_array, axes="axes")
    """
    x = np.asarray(array)
    axes = axes.lower().strip()
    if len(axes) != x.ndim:
        raise ValueError(f"axes='{axes}' does not match array ndim={x.ndim}")
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"axes must contain y and x, received '{axes}'")

    y_index = axes.index("y")
    x_index = axes.index("x")
    c_index = axes.index("c") if "c" in axes else None
    leading_indices = [
        index
        for index in range(x.ndim)
        if index not in {y_index, x_index, c_index}
    ]

    permutation: list[int] = []
    if c_index is not None:
        permutation.append(c_index)
    permutation.extend(leading_indices)
    permutation.extend([y_index, x_index])
    moved = np.transpose(x, permutation)
    if c_index is None:
        moved = moved[None, ...]

    n_channels = moved.shape[0]
    leading_shape = moved.shape[1:-2]
    n_frames = int(np.prod(leading_shape)) if leading_shape else 1
    reshaped = moved.reshape(n_channels, n_frames, moved.shape[-2], moved.shape[-1])
    frame_limit = n_frames if max_frames is None else min(n_frames, max_frames)

    for channel_index in range(n_channels):
        for frame_index in range(frame_limit):
            yield channel_index, frame_index, reshaped[channel_index, frame_index]


def channel_label(dataset: str, channel_index: int) -> str:
    """Return a stable biological/display label for a dataset channel.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        channel_index (int): Zero-based index selecting channel.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = channel_label(dataset="2d_time", channel_index=1)
    """
    dataset_key = dataset.strip().lower()
    if dataset_key == "2d_time":
        return "HADA" if channel_index == 0 else f"ch{channel_index}"
    if dataset_key == "2d_wga_dapi":
        return {0: "DAPI", 1: "WGA"}.get(channel_index, f"ch{channel_index}")
    return f"ch{channel_index}"
