"""
Optional directional notch filtering for 2D OME-Zarr images.

The filter attenuates angular wedges in the centered Fourier plane and is
retained for comparison with directional line-noise removal. It is not the
default 2D filter; the production workflow uses local-threshold filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Literal

import numpy as np

from PFT.core_prog_parts.common_paths import dataset_img_dir, find_project_root, normalize_dataset_name, results_filters_dir
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs


def _repo_root() -> Path:
    """Internal helper used by this module.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _repo_root()
    """
    return find_project_root()


def _dataset_dir(dataset: str) -> Path:
    """Supported dataset names: - "2d_time" (blue only) - "2d_wga_dapi" (2 channels) - also accepts typo aliases: "2d_dpa_wagi", "2d_dpa_wga", "2d_dpa_wagi".

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _dataset_dir(dataset="2d_time")
    """
    return dataset_img_dir(normalize_dataset_name(dataset), _repo_root())


def list_omezarr_images(dataset: str) -> list[Path]:
    """List available inputs for this workflow.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = list_omezarr_images(dataset="2d_time")
    """
    ds_dir = _dataset_dir(dataset)
    if not ds_dir.exists():
        return []
    # expected: results/img/<dataset>/<stem>/image.ome.zarr
    return sorted(ds_dir.glob("*/image.ome.zarr"))



# Notch parameterization


@dataclass
class NotchParams:
    """
    Notch filtering for line artifacts is usually implemented as angular wedges (band-stop).
    half_width_deg: wedge half width (Δθ). Start small (e.g., 2–6 degrees).
    r_min: exclude low frequencies (DC disk). In pixels in FFT plane.
    r_max: optional max radius (None -> full to edge).
    depth: attenuation strength in [0..1] where 1 = full suppression, 0.5 = partial.
    smooth: if True, use soft (Gaussian) angular roll-off; if False, hard wedge.
    """
    angles_deg: list[float]
    half_width_deg: float = 5.0
    r_min: int = 32
    r_max: int | None = None
    depth: float = 1.0
    smooth: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` when the notch definition is invalid.

        Raises:
            ValueError: If the supplied inputs or runtime state violate the function's requirements.

        Example:
            >>> instance = NotchParams(...)
            >>> instance.validate()
        """
        if not self.angles_deg:
            raise ValueError("angles_deg must contain at least one angle")
        if self.half_width_deg <= 0.0 or self.half_width_deg > 90.0:
            raise ValueError("half_width_deg must be in (0, 90]")
        if self.r_min < 0:
            raise ValueError("r_min must be >= 0")
        if self.r_max is not None and self.r_max <= self.r_min:
            raise ValueError("r_max must be greater than r_min")
        if not 0.0 <= self.depth <= 1.0:
            raise ValueError("depth must be in [0, 1]")


def _to_numpy(a) -> np.ndarray:
    # load_ome_zarr may return dask arrays; enforce numpy
    """Internal helper used by this module.

    Args:
        a (Any): Value specifying a for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _to_numpy(a=...)
    """
    try:
        import dask.array as da  # type: ignore
        if isinstance(a, da.Array):
            return a.compute()
    except Exception:
        pass
    return np.asarray(a)


def _ensure_cyx(x: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Return array where spatial dims are last (.., y, x). The writer already saves yx last, but this makes the logic robust if user provides other arrays.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        axes (str): Axis specification describing the dimensional order of the image data.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _ensure_cyx(x=image_array, axes="axes")
    """
    if "y" in axes and "x" in axes and axes.endswith("yx"):
        return x, axes
    if ("y" not in axes) or ("x" not in axes):
        raise ValueError(f"Axes missing spatial dims: axes={axes}")
    y_i, x_i = axes.index("y"), axes.index("x")
    perm = [i for i in range(x.ndim) if i not in (y_i, x_i)] + [y_i, x_i]
    x2 = np.transpose(x, perm)
    axes2 = "".join(axes[i] for i in perm)
    return x2, axes2


def _fft2_logmag(img2d: np.ndarray) -> np.ndarray:
    """Internal helper used by this module.

    Args:
        img2d (np.ndarray): Array containing img2d.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _fft2_logmag(img2d=image_array)
    """
    x = img2d.astype(np.float32, copy=False)
    x = x - float(np.mean(x))
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)
    return np.log1p(mag).astype(np.float32)


def _angle_grid(h: int, w: int) -> np.ndarray:
    """Internal helper used by this module.

    Args:
        h (int): Numerical value controlling h.
        w (int): Numerical value controlling w.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _angle_grid(h=1, w=1)
    """
    cy = float(h // 2)
    cx = float(w // 2)
    yy, xx = np.indices((h, w), dtype=np.float32)
    ang = np.degrees(np.arctan2(yy - cy, xx - cx))  # [-180..180]
    return ang


def _radius_grid(h: int, w: int) -> np.ndarray:
    """Internal helper used by this module.

    Args:
        h (int): Numerical value controlling h.
        w (int): Numerical value controlling w.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _radius_grid(h=1, w=1)
    """
    cy = float(h // 2)
    cx = float(w // 2)
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr


def build_wedge_mask(shape_hw: tuple[int, int], p: NotchParams) -> np.ndarray:
    """Build the multiplicative Fourier transfer function for a notch filter.

    The returned array equals one outside the selected angular and radial
    regions. Inside each selected wedge it is attenuated according to ``depth``
    and ``smooth``.

    Args:
        shape_hw (tuple[int, int]): Numerical value controlling shape hw.
        p (NotchParams): Filesystem path to the resource being processed.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = build_wedge_mask(shape_hw=1, p=...)
    """
    p.validate()
    h, w = shape_hw
    if h < 2 or w < 2:
        raise ValueError(f"Notch filtering requires a 2D image larger than 1 pixel, received {shape_hw}")
    ang = _angle_grid(h, w)
    rr = _radius_grid(h, w)

    r_max = p.r_max if p.r_max is not None else float(rr.max())
    valid_r = (rr >= float(p.r_min)) & (rr <= float(r_max))

    # angular distance helper: smallest absolute difference modulo 360
    def ang_dist(a: np.ndarray, a0: float) -> np.ndarray:
        """Helper function used by this module.

        Args:
            a (np.ndarray): Array containing a.
            a0 (float): Numerical value controlling a0.

        Returns:
            np.ndarray: Array containing the processed result.

        Example:
            >>> result = ang_dist(a=image_array, a0=0.5)
        """
        d = (a - a0 + 180.0) % 360.0 - 180.0
        return np.abs(d)

    H = np.ones((h, w), dtype=np.float32)

    # for each angle, suppress both theta and theta+180 automatically by using dist modulo 180 via symmetry
    angles = []
    for a in p.angles_deg:
        a = float(a)
        angles.extend([a, (a + 180.0) % 360.0])

    for a0 in angles:
        d = ang_dist(ang, a0)
        in_wedge = valid_r & (d <= float(p.half_width_deg))
        if not np.any(in_wedge):
            continue

        if p.smooth:
            # Gaussian angular roll-off (soft edges)
            # weight goes from 1 outside to (1 - depth) at center
            sigma = max(float(p.half_width_deg) / 2.0, 1e-3)
            wgt = np.exp(-(d ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
            # Limit the Gaussian to the requested angular neighborhood.
            # Outside the wedge, the transfer function remains exactly one.
            atten = 1.0 - float(p.depth) * wgt
            H = np.minimum(H, np.where(in_wedge, atten, 1.0))
        else:
            H[in_wedge] = np.minimum(H[in_wedge], 1.0 - float(p.depth))

    return H


def apply_notch_filter_2d(img2d: np.ndarray, p: NotchParams) -> tuple[np.ndarray, np.ndarray]:
    """Apply directional Fourier attenuation to one ``(Y, X)`` image plane.

    Returns the filtered ``float32`` image and the transfer function used.

    Args:
        img2d (np.ndarray): Array containing img2d.
        p (NotchParams): Filesystem path to the resource being processed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = apply_notch_filter_2d(img2d=image_array, p=...)
    """
    p.validate()
    x = np.asarray(img2d, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D image plane, received shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Notch-filter input contains NaN or infinite values")
    mu = float(np.mean(x))
    x0 = x - mu

    F = np.fft.fftshift(np.fft.fft2(x0))
    H = build_wedge_mask(x.shape, p)
    Ff = F * H
    y = np.fft.ifft2(np.fft.ifftshift(Ff))
    out = np.real(y).astype(np.float32) + mu
    return out, H



def run_notch_on_dataset(
    dataset: str,
    params: NotchParams,
    *,
    apply: bool,
    channel_mode: Literal["auto", "blue", "green"] = "auto",
    image_index: int | None = None,
    out_subdir_name: str | None = None,
) -> Path:
    """Apply the optional notch filter to one selected OME-Zarr image.

    ``apply=False`` performs dataset discovery and output preparation without
    changing pixel values. The function returns the output directory.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        params (NotchParams): Value specifying params for the operation.
        apply (bool): Boolean flag controlling apply.
        channel_mode (Literal["auto", "blue", "green"]): Value specifying channel mode for the operation. Defaults to ``"auto"``.
        image_index (int | None): Zero-based index selecting image. ``None`` selects the function's default behavior.
        out_subdir_name (str | None): Text value specifying out subdir name. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = run_notch_on_dataset(
        ...     dataset="2d_time",
        ...     params=...,
        ...     apply=True,
        ... )
    """
    zarrs = list_omezarr_images(dataset)
    if not zarrs:
        raise FileNotFoundError(
            f"No OME-Zarr images found for dataset={dataset}. "
            f"Looked in: {_dataset_dir(dataset)}"
        )

    if image_index is None:
        image_index = 0
    image_index = int(np.clip(image_index, 0, len(zarrs) - 1))
    in_path = zarrs[image_index]
    stem = in_path.parent.name

    arr, axes = load_ome_zarr(in_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)

    # decide channel selection
    # expected axes: "cyx" or "tcyx"
    if "c" in axes:
        c_i = axes.index("c")
        n_c = x.shape[c_i]
    else:
        n_c = 1

    def _pick_channels() -> list[int]:
        """Internal helper used by this module.

        Returns:
            list[int]: Collection containing the generated or selected values.

        Example:
            >>> result = _pick_channels()
        """
        if channel_mode == "blue":
            return [0]
        if channel_mode == "green":
            return [1] if n_c > 1 else [0]
        # auto:
        if dataset.strip().lower() == "2d_time":
            return [0]
        return list(range(min(n_c, 2))) if n_c > 1 else [0]

    chs = _pick_channels()

    # apply per channel
    y = x.astype(np.float32, copy=True)

    if apply:
        for c in chs:
            # slice channel plane robustly
            sl = [slice(None)] * y.ndim
            if "c" in axes:
                sl[axes.index("c")] = c
            # handle optional time by leaving as slice(None) (apply per frame)
            plane = y[tuple(sl)]
            if plane.ndim == 2:
                filt, _ = apply_notch_filter_2d(plane, params)
                y[tuple(sl)] = filt
            elif plane.ndim == 3 and "t" in axes:
                # t,y,x
                for t in range(plane.shape[0]):
                    filt, _ = apply_notch_filter_2d(plane[t], params)
                    plane[t] = filt
                y[tuple(sl)] = plane
            else:
                raise ValueError(f"Unexpected plane ndim={plane.ndim} for axes={axes}")
    else:
        # dry run: no changes
        pass

    out_root = results_filters_dir() / "Notch" / dataset
    if out_subdir_name:
        out_root = out_root / out_subdir_name
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal meta for saving
    meta = SimpleNamespace(
        pixel_size_um_x=1.0,
        pixel_size_um_y=1.0,
        pixel_size_um_z=1.0,
        channel_names=["ch0", "ch1", "ch2"][: n_c],
        source_path=str(in_path),
        axes=axes,
    )

    
    save_ome_zarr_next_to_outputs(
        out_dir,
        y,
        meta,
        overwrite=True,
        pyramid_3d=False,   # 2D datasets -> single scale
        pyramid_max_layer=0
    )

    return out_dir