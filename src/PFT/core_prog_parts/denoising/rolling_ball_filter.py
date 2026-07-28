"""Rolling-ball background subtraction for 2D fluorescence images.

The rolling-ball filter is retained as an optional, callable preprocessing
method. It is not the default 2D filter in the PFT workflow. The active default
is the local-threshold filter implemented in ``denoiser.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from skimage.restoration import rolling_ball as _skimage_rolling_ball
except Exception:  # pragma: no cover - dependency error is reported at runtime
    _skimage_rolling_ball = None


@dataclass(frozen=True)
class RollingBallParams:
    """Parameters for 2D rolling-ball background estimation."""

    radius: int = 50

    def validate(self) -> None:
        """Raise ``ValueError`` if the radius cannot define a background scale."""
        if self.radius < 1:
            raise ValueError("Rolling-ball radius must be >= 1 pixel")


def apply_rolling_ball_2d(
    image: np.ndarray,
    params: RollingBallParams | None = None,
    *,
    radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and subtract a smooth background from one 2D image.

    Returns
    -------
    corrected, background:
        Two ``float32`` arrays with the same shape as the input. Negative values
        after subtraction are clipped to zero.
    """
    if _skimage_rolling_ball is None:
        raise RuntimeError(
            "Rolling-ball filtering requires scikit-image. Install it with "
            "'pip install scikit-image'."
        )
    chosen = params or RollingBallParams(radius=50 if radius is None else int(radius))
    if radius is not None and params is not None:
        raise ValueError("Pass either params or radius, not both")
    chosen.validate()

    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D image plane, received shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Rolling-ball input contains NaN or infinite values")

    background = np.asarray(
        _skimage_rolling_ball(x, radius=chosen.radius),
        dtype=np.float32,
    )
    corrected = np.clip(x - background, 0.0, None).astype(np.float32)
    return corrected, background
