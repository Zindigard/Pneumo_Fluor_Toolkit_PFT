"""
Create 2D maximum-intensity projections from PFT 3D OME-Zarr volumes.

The workflow supports raw and Richardson-Lucy-deconvolved MIPs, either
unmasked or followed by the existing per-volume 2.5D U-Net prediction.
Deconvolution iterations and outside-mask suppression are configurable.
For example, 99.8% suppression retains 0.2% of the original outside-mask
intensity instead of setting it to zero.

All 40 Z planes are projected. The saved quantitative image remains in the
source intensity scale: raw products retain the original dtype, whereas the
selected deconvolution product is stored as float32. Display normalization is
used only for the merged-RGB QC figure. Every QC panel receives a calibrated
2 µm scale bar by default. ROI SNR is reported for every wavelength channel
and for a non-normalized merged scalar image defined as the root-sum-square
of all channels. The predicted 2.5D foreground mask defines signal/background.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from PFT.core_prog_parts.common_paths import (
    find_project_root,
    project_relative_path,
    resolve_project_path,
)
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.metadata_3d import (
    coordinate_scale_for_level,
    copyable_root_metadata,
    resolve_channel_optics,
)
from PFT.core_prog_parts.denoising.validation_3d import (
    relative_volume_path,
    target_slice_for_volume,
    volume_key,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
MIPMode = Literal["raw_unmasked", "raw_masked", "deconv_unmasked", "deconv_masked"]

DECONV_MODEL = "BW"
DECONV_DEFAULT_ITERS_BLUE = 3
DECONV_DEFAULT_ITERS_GREEN = 3
DECONV_DEFAULT_ITERS_RED = 2
DECONV_SOURCE_LEVEL = 0
EXPECTED_Z_COUNT = 40

MODE_DIRECTORY_NAMES: dict[MIPMode, str] = {
    "raw_unmasked": "raw_mip",
    "raw_masked": "raw_mip_masked",
    "deconv_unmasked": "deconv_mip",
    "deconv_masked": "deconv_mip_masked",
}


@dataclass
class MIP3DConfig:
    """Configuration for one raw source volume and one requested MIP product."""

    project_root: Path = find_project_root(Path(__file__).resolve())
    raw_zarr: Path | None = None
    mode: MIPMode = "raw_unmasked"
    raw_root: Path | None = None
    deconv_root: Path | None = None
    mask_root: Path | None = None
    output_root: Path | None = None
    level: int = 0
    expected_z_count: int = EXPECTED_Z_COUNT
    overwrite: bool = True
    qc_percentile_low: float = 1.0
    qc_percentile_high: float = 99.8
    qc_max_dimension: int = 1400
    scale_bar_um: float = 2.0
    snr_epsilon: float = 1e-12
    deconv_iters_blue: int = DECONV_DEFAULT_ITERS_BLUE
    deconv_iters_green: int = DECONV_DEFAULT_ITERS_GREEN
    deconv_iters_red: int = DECONV_DEFAULT_ITERS_RED
    outside_suppression_percent: float = 100.0


@dataclass(frozen=True)
class MIP3DOutput:
    """Saved files and essential provenance for one MIP product."""

    sample: str
    mode: str
    raw_zarr: Path
    projection_source_zarr: Path
    mask_zarr: Path | None
    portable_unmasked_mip_zarr: Path | None
    portable_mask_zarr: Path | None
    output_zarr: Path
    qc_png: Path
    report_json: Path
    snr_csv: Path | None
    snr_json: Path | None
    target_slice_1based: int
    output_dtype: str
    output_shape_cyx: tuple[int, int, int]


@dataclass(frozen=True)
class MIPSNRRow:
    """ROI-SNR comparison for one channel or the merged RSS image."""

    sample: str
    target_slice_1based: int
    channel_index: int
    channel_name: str
    signal_pixels: int
    background_pixels: int
    raw_target_signal_mean: float
    raw_target_background_mean: float
    raw_target_background_sd: float
    raw_target_snr: float
    mip_before_signal_mean: float
    mip_before_background_mean: float
    mip_before_background_sd: float
    mip_before_snr: float
    delta_mip_before_vs_raw_target: float
    masked_mip_signal_mean: float
    masked_mip_background_mean: float
    masked_mip_background_sd: float
    masked_mip_snr: float | None
    masked_mip_snr_status: str


def _roi_snr_components(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    epsilon: float = 1e-12,
) -> tuple[int, int, float, float, float, float]:
    """Use the current PFT ROI-SNR formula without modifying stored intensities.

    Args:
        image (np.ndarray): Input image array to process.
        foreground_mask (np.ndarray): Array containing foreground mask.
        epsilon (float): Numerical value controlling epsilon. Defaults to ``1e-12``.

    Returns:
        tuple[int, int, float, float, float, float]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _roi_snr_components(image=image_array, foreground_mask=image_array)
    """
    values = np.asarray(image, dtype=np.float64)
    mask = np.asarray(foreground_mask, dtype=bool)
    if values.ndim != 2 or mask.ndim != 2 or values.shape != mask.shape:
        raise ValueError(
            f"ROI SNR expects matching 2D image/mask, got {values.shape} and {mask.shape}"
        )
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    finite = np.isfinite(values)
    signal = values[mask & finite]
    background = values[(~mask) & finite]
    if signal.size < 1:
        raise ValueError("ROI SNR requires at least one finite foreground pixel")
    if background.size < 2:
        raise ValueError("ROI SNR requires at least two finite background pixels")
    signal_mean = float(np.mean(signal))
    background_mean = float(np.mean(background))
    background_sd = float(np.std(background, ddof=1))
    snr = float((signal_mean - background_mean) / (background_sd + epsilon))
    return (
        int(signal.size),
        int(background.size),
        signal_mean,
        background_mean,
        background_sd,
        snr,
    )


RGB_REFERENCE_WAVELENGTHS_NM: dict[str, float] = {
    "red": 561.0,
    "green": 488.0,
    "blue": 405.0,
}


def resolve_rgb_source_channels(image_zarr: Path, *, level: int = 0) -> tuple[int, int, int]:
    """Resolve source channels in red, green, blue order from wavelength metadata.

    Args:
        image_zarr (Path): Filesystem path used for image Zarr.
        level (int): Numerical value controlling level. Defaults to ``0``.

    Returns:
        tuple[int, int, int]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_rgb_source_channels(image_zarr=Path("path/to/resource"))
    """
    assignments: dict[str, int] = {}
    for item in resolve_channel_optics(image_zarr, level=level):
        colour, reference_nm = min(
            RGB_REFERENCE_WAVELENGTHS_NM.items(),
            key=lambda pair: abs(float(item.wavelength_nm) - pair[1]),
        )
        if abs(float(item.wavelength_nm) - reference_nm) > 35.0:
            raise ValueError(
                f"Channel {item.index} ({item.name}) at {item.wavelength_nm:g} nm "
                "cannot be mapped safely to 405/488/561 nm RGB."
            )
        if colour in assignments:
            raise ValueError(f"Multiple source channels map to {colour}")
        assignments[colour] = int(item.index)
    missing = [colour for colour in ("red", "green", "blue") if colour not in assignments]
    if missing:
        raise ValueError(f"Missing wavelength channel(s) for merged RGB display: {missing}")
    return assignments["red"], assignments["green"], assignments["blue"]


def _require_mode(mode: str) -> MIPMode:
    """Return require mode for the supplied inputs.

    Args:
        mode (str): Text value specifying mode.

    Returns:
        MIPMode: Result produced by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _require_mode(mode="mode")
    """
    if mode not in MODE_DIRECTORY_NAMES:
        raise ValueError(
            f"Unknown MIP mode {mode!r}; expected one of {tuple(MODE_DIRECTORY_NAMES)}"
        )
    return mode  # type: ignore[return-value]


def _open_level_array(zarr_path: Path, level: int) -> tuple[Any, str, dict[str, Any]]:
    """Open one OME-Zarr level lazily and return its array, axes, and metadata.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.
        level (int): Numerical value controlling level.

    Returns:
        tuple[Any, str, dict[str, Any]]: Mapping containing the generated or resolved values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ImportError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _open_level_array(zarr_path=Path("path/to/resource"), level=1)
    """
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "MIP creation requires zarr and ome-zarr from the project environment."
        ) from exc

    zarr_path = Path(zarr_path).expanduser().resolve()
    if not zarr_path.is_dir() or not zarr_path.name.endswith(".ome.zarr"):
        raise FileNotFoundError(f"Expected an OME-Zarr directory: {zarr_path}")

    metadata = extract_ome_zarr_meta_for_compare(zarr_path, level=level)
    axes = str(metadata.get("axes") or "").lower()
    array_path = str(metadata.get("array_path") or level)
    root = zarr.open_group(str(zarr_path), mode="r")
    array = root[array_path]
    if len(axes) != array.ndim:
        raise ValueError(
            f"OME axes {axes!r} do not match shape {tuple(array.shape)} for {zarr_path}"
        )
    return array, axes, metadata


class _RemovedSingletonAxisView:
    """Read-only array view that removes one singleton axis without loading it."""

    def __init__(self, base: Any, removed_axis: int) -> None:
        """Initialize a ``_RemovedSingletonAxisView`` instance.

        Args:
            base (Any): Value specifying base for the operation.
            removed_axis (int): Numerical value controlling removed axis.

        Example:
            >>> instance = _RemovedSingletonAxisView(base=..., removed_axis=1)
        """
        self._base = base
        self._removed_axis = int(removed_axis)
        self.shape = tuple(
            int(value)
            for index, value in enumerate(base.shape)
            if index != self._removed_axis
        )
        self.ndim = len(self.shape)
        self.dtype = np.dtype(base.dtype)

    def __getitem__(self, key: Any) -> Any:
        """Return the item associated with the supplied key or index.

        Args:
            key (Any): Key used to access or identify an entry in a mapping.

        Returns:
            Any: Result produced by the operation.

        Raises:
            IndexError: If the supplied inputs or runtime state violate the function's requirements.

        Example:
            >>> instance = _RemovedSingletonAxisView(...)
            >>> result = instance.__getitem__(key=...)
        """
        if not isinstance(key, tuple):
            key = (key,)
        key_items = list(key)
        if Ellipsis in key_items:
            ellipsis_index = key_items.index(Ellipsis)
            missing = self.ndim - (len(key_items) - 1)
            key_items = (
                key_items[:ellipsis_index]
                + [slice(None)] * missing
                + key_items[ellipsis_index + 1 :]
            )
        if len(key_items) < self.ndim:
            key_items.extend([slice(None)] * (self.ndim - len(key_items)))
        if len(key_items) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices for singleton-axis view, got {len(key_items)}"
            )
        key_items.insert(self._removed_axis, 0)
        return self._base[tuple(key_items)]


def _canonicalize_czyx_view(array: Any, axes: str, source: Path) -> tuple[Any, str]:
    """Return a lazy CZYX view and remove a singleton time axis when present.

    Args:
        array (Any): Value specifying array for the operation.
        axes (str): Axis specification describing the dimensional order of the image data.
        source (Path): Filesystem path used for source.

    Returns:
        tuple[Any, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _canonicalize_czyx_view(
        ...     array=image_array,
        ...     axes="axes",
        ...     source=Path("path/to/resource"),
        ... )
    """
    current_axes = str(axes).lower()
    if "t" in current_axes:
        time_axis = current_axes.index("t")
        time_count = int(array.shape[time_axis])
        if time_count != 1:
            raise ValueError(
                "A single 3D volume is required, but the OME-Zarr contains "
                f"T={time_count} time points with axes={current_axes!r}: {source}"
            )
        array = _RemovedSingletonAxisView(array, time_axis)
        current_axes = current_axes[:time_axis] + current_axes[time_axis + 1 :]

    if current_axes != "czyx":
        raise ValueError(
            f"Expected CZYX or singleton-time CZYX OME-Zarr, "
            f"received axes={axes!r}: {source}"
        )
    return array, current_axes


def _require_czyx(array: Any, axes: str, source: Path, expected_z_count: int) -> None:
    """Validate the thesis 3D source representation without loading the volume.

    Args:
        array (Any): Value specifying array for the operation.
        axes (str): Axis specification describing the dimensional order of the image data.
        source (Path): Filesystem path used for source.
        expected_z_count (int): Number of expected z used by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> _require_czyx(
        ...     array=image_array,
        ...     axes="axes",
        ...     source=Path("path/to/resource"),
        ...     expected_z_count=1,
        ... )
    """
    if axes != "czyx":
        raise ValueError(f"Expected CZYX OME-Zarr, received axes={axes!r}: {source}")
    if int(array.shape[0]) != 3:
        raise ValueError(f"Expected three fluorescence channels, got shape={array.shape}: {source}")
    if int(array.shape[1]) != int(expected_z_count):
        raise ValueError(
            f"Expected exactly {expected_z_count} Z planes, got Z={array.shape[1]}: {source}"
        )
    if int(array.shape[2]) < 1 or int(array.shape[3]) < 1:
        raise ValueError(f"Invalid spatial shape {array.shape}: {source}")


def maximum_intensity_projection_cyx(array_czyx: Any) -> np.ndarray:
    """Calculate a C×Y×X maximum projection while loading only one Z plane at a time.

    Args:
        array_czyx (Any): Value specifying array czyx for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = maximum_intensity_projection_cyx(array_czyx=...)
    """
    if getattr(array_czyx, "ndim", None) != 4:
        raise ValueError(f"Expected a four-dimensional CZYX array, got {array_czyx.shape}")
    c_count, z_count, y_count, x_count = (int(value) for value in array_czyx.shape)
    if z_count < 1:
        raise ValueError("Cannot create a MIP from an empty Z axis")

    output = np.empty((c_count, y_count, x_count), dtype=np.dtype(array_czyx.dtype))
    for channel in range(c_count):
        channel_mip = np.asarray(array_czyx[channel, 0, :, :]).copy()
        if not np.isfinite(channel_mip).all():
            raise ValueError(f"Non-finite values in channel {channel}, Z1")
        for z_index in range(1, z_count):
            plane = np.asarray(array_czyx[channel, z_index, :, :])
            if not np.isfinite(plane).all():
                raise ValueError(
                    f"Non-finite values in channel {channel}, Z{z_index + 1}"
                )
            np.maximum(channel_mip, plane, out=channel_mip)
        output[channel] = channel_mip
    return output


def _read_cyx_slice(array_czyx: Any, slice_1based: int) -> np.ndarray:
    """Read one target slice from a CZYX array without changing its stored scale.

    Args:
        array_czyx (Any): Value specifying array czyx for the operation.
        slice_1based (int): Numerical value controlling slice 1based.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        IndexError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _read_cyx_slice(array_czyx=..., slice_1based=1)
    """
    z_index = int(slice_1based) - 1
    if not 0 <= z_index < int(array_czyx.shape[1]):
        raise IndexError(
            f"Target Z{slice_1based} is outside a stack with Z={array_czyx.shape[1]}"
        )
    plane = np.asarray(array_czyx[:, z_index, :, :])
    if not np.isfinite(plane).all():
        raise ValueError(f"Non-finite values in configured raw target Z{slice_1based}")
    return plane


def _validate_iteration_count(value: int, name: str) -> int:
    """Validate iteration count against the required constraints.

    Args:
        value (int): Value to validate, transform, store, or forward.
        name (str): Name used to identify the current object, resource, or output.

    Returns:
        int: Computed numerical result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _validate_iteration_count(value=1, name="name")
    """
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return result


def deconvolution_iteration_tag(blue: int, green: int, red: int) -> str:
    """Return the folder tag used by the 3D deconvolution workflow.

    Args:
        blue (int): Numerical value controlling blue.
        green (int): Numerical value controlling green.
        red (int): Numerical value controlling red.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = deconvolution_iteration_tag(
        ...     blue=1,
        ...     green=1,
        ...     red=1,
        ... )
    """
    blue = _validate_iteration_count(blue, "blue iterations")
    green = _validate_iteration_count(green, "green iterations")
    red = _validate_iteration_count(red, "red iterations")
    return f"iterB{blue}_G{green}_R{red}"


def resolve_deconvolved_zarr(
    raw_zarr: Path,
    deconv_root: Path,
    *,
    iters_blue: int = DECONV_DEFAULT_ITERS_BLUE,
    iters_green: int = DECONV_DEFAULT_ITERS_GREEN,
    iters_red: int = DECONV_DEFAULT_ITERS_RED,
) -> Path:
    """Resolve one exact BW Richardson-Lucy result for a configured raw volume.

    Args:
        raw_zarr (Path): Filesystem path used for raw Zarr.
        deconv_root (Path): Directory used for deconv.
        iters_blue (int): Numerical value controlling iters blue. Defaults to ``DECONV_DEFAULT_ITERS_BLUE``.
        iters_green (int): Numerical value controlling iters green. Defaults to ``DECONV_DEFAULT_ITERS_GREEN``.
        iters_red (int): Numerical value controlling iters red. Defaults to ``DECONV_DEFAULT_ITERS_RED``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_deconvolved_zarr(raw_zarr=Path("path/to/resource"), deconv_root=Path("path/to/resource"))
    """
    relative = relative_volume_path(raw_zarr)
    sample = relative.name
    iteration_tag = deconvolution_iteration_tag(
        iters_blue, iters_green, iters_red
    )
    result = (
        Path(deconv_root)
        / relative.parent
        / (
            f"{sample}__SK_RL__PSF{DECONV_MODEL}__{iteration_tag}"
            f"__sourceL{DECONV_SOURCE_LEVEL}"
        )
        / "image.ome.zarr"
    )
    if not result.is_dir():
        raise FileNotFoundError(
            "Required deconvolution result is missing. Expected: " f"{result}"
        )
    return result.resolve()


def resolve_deconvolved_332_zarr(raw_zarr: Path, deconv_root: Path) -> Path:
    """Backward-compatible resolver for the original fixed 3/3/2 product.

    Args:
        raw_zarr (Path): Filesystem path used for raw Zarr.
        deconv_root (Path): Directory used for deconv.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = resolve_deconvolved_332_zarr(raw_zarr=Path("path/to/resource"), deconv_root=Path("path/to/resource"))
    """
    return resolve_deconvolved_zarr(
        raw_zarr,
        deconv_root,
        iters_blue=3,
        iters_green=3,
        iters_red=2,
    )


def resolve_predicted_mask_zarr(raw_zarr: Path, mask_root: Path) -> Path:
    """Resolve the existing per-volume 2.5D U-Net mask for one raw source.

    Args:
        raw_zarr (Path): Filesystem path used for raw Zarr.
        mask_root (Path): Directory used for mask.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_predicted_mask_zarr(raw_zarr=Path("path/to/resource"), mask_root=Path("path/to/resource"))
    """
    result = Path(mask_root) / relative_volume_path(raw_zarr) / "pred_mask.ome.zarr"
    if not result.is_dir():
        raise FileNotFoundError(
            "The existing 2.5D U-Net prediction is missing. Run 2.5D inference first: "
            f"{result}"
        )
    return result.resolve()


def _load_target_mask_yx(
    mask_zarr: Path,
    *,
    target_slice_1based: int,
    expected_yx: tuple[int, int],
) -> np.ndarray:
    """Read the target plane from a YX or broadcast ZYX binary mask OME-Zarr.

    Args:
        mask_zarr (Path): Filesystem path used for mask Zarr.
        target_slice_1based (int): Numerical value controlling target slice 1based.
        expected_yx (tuple[int, int]): Numerical value controlling expected yx.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        IndexError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_target_mask_yx(
        ...     mask_zarr=Path("path/to/resource"),
        ...     target_slice_1based=1,
        ...     expected_yx=1,
        ... )
    """
    array, axes, _metadata = _open_level_array(mask_zarr, level=0)
    if axes == "yx":
        mask = np.asarray(array)
    elif axes == "zyx":
        z_index = int(target_slice_1based) - 1
        if not 0 <= z_index < int(array.shape[0]):
            raise IndexError(
                f"Mask does not contain target Z{target_slice_1based}: shape={array.shape}"
            )
        mask = np.asarray(array[z_index, :, :])
    else:
        raise ValueError(f"Expected YX or ZYX predicted mask, received axes={axes!r}: {mask_zarr}")

    if tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(
            f"Predicted-mask/image mismatch: mask={mask.shape}, expected={expected_yx}: {mask_zarr}"
        )
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary) or np.all(binary):
        raise ValueError(f"Predicted mask must contain foreground and background: {mask_zarr}")
    return binary


def apply_background_suppression_mask(
    mip_cyx: np.ndarray,
    mask_yx: np.ndarray,
    *,
    suppression_percent: float,
) -> np.ndarray:
    """Suppress intensities outside the mask while preserving the source dtype.

    ``suppression_percent=99.8`` retains 0.2% of each outside-mask value.
    ``suppression_percent=100`` sets the outside-mask values exactly to zero.

    Args:
        mip_cyx (np.ndarray): Array containing maximum-intensity projection cyx.
        mask_yx (np.ndarray): Array containing mask yx.
        suppression_percent (float): Numerical value controlling suppression percent.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = apply_background_suppression_mask(
        ...     mip_cyx=image_array,
        ...     mask_yx=image_array,
        ...     suppression_percent=0.5,
        ... )
    """
    if mip_cyx.ndim != 3:
        raise ValueError(f"Expected CYX MIP, got shape={mip_cyx.shape}")
    if tuple(mip_cyx.shape[1:]) != tuple(mask_yx.shape):
        raise ValueError(f"MIP/mask shape mismatch: {mip_cyx.shape} vs {mask_yx.shape}")
    suppression = float(suppression_percent)
    if not np.isfinite(suppression) or not 0.0 <= suppression <= 100.0:
        raise ValueError(
            "outside suppression must be between 0 and 100 percent, "
            f"got {suppression_percent}"
        )
    retain_fraction = 1.0 - suppression / 100.0
    mask = np.asarray(mask_yx) > 0
    if suppression == 100.0:
        return np.where(mask[None, :, :], mip_cyx, 0).astype(
            mip_cyx.dtype, copy=False
        )

    values = np.asarray(mip_cyx)
    output = values.copy()
    outside = ~mask
    if np.issubdtype(values.dtype, np.integer):
        scaled = np.rint(
            values[:, outside].astype(np.float64) * retain_fraction
        )
        limits = np.iinfo(values.dtype)
        output[:, outside] = np.clip(scaled, limits.min, limits.max).astype(
            values.dtype
        )
    else:
        output[:, outside] = (
            values[:, outside] * np.asarray(retain_fraction, dtype=values.dtype)
        )
    return output


def apply_zero_background_mask(mip_cyx: np.ndarray, mask_yx: np.ndarray) -> np.ndarray:
    """Backward-compatible exact-zero mask application.

    Args:
        mip_cyx (np.ndarray): Array containing maximum-intensity projection cyx.
        mask_yx (np.ndarray): Array containing mask yx.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = apply_zero_background_mask(mip_cyx=image_array, mask_yx=image_array)
    """
    return apply_background_suppression_mask(
        mip_cyx, mask_yx, suppression_percent=100.0
    )


def _cyx_coordinate_scale(source_zarr: Path, *, level: int) -> list[float]:
    """Drop the projected Z scale while preserving C, Y, and X sampling.

    Args:
        source_zarr (Path): Filesystem path used for source Zarr.
        level (int): Numerical value controlling level.

    Returns:
        list[float]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _cyx_coordinate_scale(source_zarr=Path("path/to/resource"), level=1)
    """
    metadata = extract_ome_zarr_meta_for_compare(source_zarr, level=level)
    axes = str(metadata.get("axes") or "").lower()
    scale = coordinate_scale_for_level(source_zarr, level=level)
    if len(scale) != len(axes):
        raise ValueError(f"Coordinate scale {scale} does not match axes {axes!r}")
    if not all(axis in axes for axis in "cyx"):
        raise ValueError(f"Cannot derive CYX scale from axes={axes!r}")
    return [float(scale[axes.index(axis)]) for axis in "cyx"]


def _to_rgb_hwc(cyx: np.ndarray, rgb_source_channels: Sequence[int]) -> np.ndarray:
    """Map source channels to red/green/blue display order without normalization.

    Args:
        cyx (np.ndarray): Array containing cyx.
        rgb_source_channels (Sequence[int]): Numerical value controlling RGB representation source channels.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _to_rgb_hwc(cyx=image_array, rgb_source_channels=1)
    """
    if len(rgb_source_channels) != 3:
        raise ValueError(f"Expected three RGB source indices, got {rgb_source_channels}")
    return np.stack(
        [np.asarray(cyx[int(channel)], dtype=np.float32) for channel in rgb_source_channels],
        axis=-1,
    )


def _raw_scale_rgb(
    cyx_images: Sequence[np.ndarray],
    rgb_source_channels: Sequence[int],
    percentile_high: float,
) -> list[np.ndarray]:
    """Create merged RGB views using one shared zero-based scale per colour.

    Args:
        cyx_images (Sequence[np.ndarray]): Array containing cyx images.
        rgb_source_channels (Sequence[int]): Numerical value controlling RGB representation source channels.
        percentile_high (float): Numerical value controlling percentile high.

    Returns:
        list[np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = _raw_scale_rgb(
        ...     cyx_images=image_array,
        ...     rgb_source_channels=1,
        ...     percentile_high=0.5,
        ... )
    """
    rgb_images = [_to_rgb_hwc(image, rgb_source_channels) for image in cyx_images]
    high_limits: list[float] = []
    for rgb_channel in range(3):
        candidate_limits = [
            float(np.percentile(image[..., rgb_channel], percentile_high))
            for image in rgb_images
        ]
        high = max(candidate_limits)
        high_limits.append(high if np.isfinite(high) and high > 0 else 1.0)

    outputs: list[np.ndarray] = []
    for image in rgb_images:
        normalized = np.empty_like(image, dtype=np.float32)
        for rgb_channel, high in enumerate(high_limits):
            normalized[..., rgb_channel] = np.clip(
                image[..., rgb_channel] / high,
                0.0,
                1.0,
            )
        outputs.append(normalized)
    return outputs


def _independently_normalized_rgb(
    cyx: np.ndarray,
    rgb_source_channels: Sequence[int],
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    """Percentile-normalize each colour independently for display only.

    Args:
        cyx (np.ndarray): Array containing cyx.
        rgb_source_channels (Sequence[int]): Numerical value controlling RGB representation source channels.
        percentile_low (float): Numerical value controlling percentile low.
        percentile_high (float): Numerical value controlling percentile high.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _independently_normalized_rgb(
        ...     cyx=image_array,
        ...     rgb_source_channels=1,
        ...     percentile_low=0.5,
        ...     percentile_high=0.5,
        ... )
    """
    rgb = _to_rgb_hwc(cyx, rgb_source_channels)
    output = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[..., channel]
        low = float(np.percentile(plane, percentile_low))
        high = float(np.percentile(plane, percentile_high))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            output[..., channel] = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return output


def _downsample_cyx_for_qc(
    cyx: np.ndarray, max_dimension: int
) -> tuple[np.ndarray, int]:
    """Return a display-only CYX array and its integer spatial stride.

    Args:
        cyx (np.ndarray): Array containing cyx.
        max_dimension (int): Maximum permitted value of dimension.

    Returns:
        tuple[np.ndarray, int]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _downsample_cyx_for_qc(cyx=image_array, max_dimension=1)
    """
    if max_dimension < 256:
        raise ValueError("qc_max_dimension must be at least 256 pixels")
    largest = max(int(cyx.shape[-2]), int(cyx.shape[-1]))
    stride = max(1, int(np.ceil(largest / float(max_dimension))))
    return cyx[:, ::stride, ::stride], stride


def _x_pixel_size_um(source_zarr: Path, *, level: int) -> float:
    """Read the physical X sampling in micrometres from OME-Zarr metadata.

    Args:
        source_zarr (Path): Filesystem path used for source Zarr.
        level (int): Numerical value controlling level.

    Returns:
        float: Computed numerical result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _x_pixel_size_um(source_zarr=Path("path/to/resource"), level=1)
    """
    metadata = extract_ome_zarr_meta_for_compare(source_zarr, level=level)
    axes = str(metadata.get("axes") or "").lower()
    voxel_size_um = metadata.get("voxel_size_um")
    if "x" not in axes or not isinstance(voxel_size_um, dict):
        raise ValueError(
            f"Physical X sampling is missing from OME-Zarr metadata: axes={axes!r}, "
            f"voxel_size_um={voxel_size_um!r}"
        )
    raw_value = voxel_size_um.get("x")
    if raw_value is None:
        raise ValueError(f"voxel_size_um.x is missing in {source_zarr}")
    value = float(raw_value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid X pixel size {value} µm in {source_zarr}")
    return value


def _add_scale_bar(
    axis: Any,
    *,
    image_shape_yx: tuple[int, int],
    x_pixel_size_um: float,
    display_stride: int,
    length_um: float,
) -> None:
    """Draw a physically calibrated scale bar in the lower-right corner.

    Args:
        axis (Any): Array axis along which the operation is performed.
        image_shape_yx (tuple[int, int]): Numerical value controlling image shape yx.
        x_pixel_size_um (float): Numerical value controlling x pixel size um.
        display_stride (int): Numerical value controlling display stride.
        length_um (float): Numerical value controlling length um.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> _add_scale_bar(
        ...     axis=...,
        ...     image_shape_yx=1,
        ...     x_pixel_size_um=0.5,
        ...     display_stride=1,
        ...     length_um=0.5,
        ... )
    """
    if not np.isfinite(length_um) or length_um <= 0:
        raise ValueError(f"scale-bar length must be positive, got {length_um}")
    if display_stride < 1:
        raise ValueError(f"display_stride must be >=1, got {display_stride}")
    height, width = (int(v) for v in image_shape_yx)
    bar_pixels = float(length_um) / (float(x_pixel_size_um) * float(display_stride))
    if bar_pixels <= 1 or bar_pixels >= width * 0.8:
        raise ValueError(
            f"A {length_um:g} µm scale bar is not suitable for width={width}, "
            f"x_pixel_size={x_pixel_size_um:g} µm, stride={display_stride}"
        )
    x_end = width * 0.94
    x_start = x_end - bar_pixels
    y = height * 0.92
    axis.plot([x_start, x_end], [y, y], color="black", linewidth=6, solid_capstyle="butt")
    axis.plot([x_start, x_end], [y, y], color="white", linewidth=3, solid_capstyle="butt")
    label = axis.text(
        (x_start + x_end) / 2.0,
        y - height * 0.025,
        f"{length_um:g} µm",
        color="white",
        fontsize=9,
        ha="center",
        va="bottom",
    )
    label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])


def _save_merged_qc(
    *,
    raw_target_cyx: np.ndarray,
    mip_before_cyx: np.ndarray,
    mip_output_cyx: np.ndarray,
    rgb_source_channels: Sequence[int],
    target_slice_1based: int,
    sample: str,
    mode: MIPMode,
    output_png: Path,
    outside_suppression_percent: float,
    percentile_low: float,
    percentile_high: float,
    max_dimension: int,
    x_pixel_size_um: float,
    scale_bar_um: float,
) -> Path:
    """Save merged-only raw-scale and normalized MIP comparisons.

    Args:
        raw_target_cyx (np.ndarray): Array containing raw target cyx.
        mip_before_cyx (np.ndarray): Array containing maximum-intensity projection before cyx.
        mip_output_cyx (np.ndarray): Array containing maximum-intensity projection output cyx.
        rgb_source_channels (Sequence[int]): Numerical value controlling RGB representation source channels.
        target_slice_1based (int): Numerical value controlling target slice 1based.
        sample (str): Text value specifying sample.
        mode (MIPMode): Value specifying mode for the operation.
        output_png (Path): Filesystem path used for output PNG image.
        outside_suppression_percent (float): Numerical value controlling outside suppression percent.
        percentile_low (float): Numerical value controlling percentile low.
        percentile_high (float): Numerical value controlling percentile high.
        max_dimension (int): Maximum permitted value of dimension.
        x_pixel_size_um (float): Numerical value controlling x pixel size um.
        scale_bar_um (float): Numerical value controlling scale bar um.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _save_merged_qc(
        ...     raw_target_cyx=image_array,
        ...     mip_before_cyx=image_array,
        ...     mip_output_cyx=image_array,
        ...     rgb_source_channels=1,
        ...     target_slice_1based=1,
        ...     sample="sample",
        ...     mode=...,
        ...     output_png=Path("path/to/resource"),
        ...     outside_suppression_percent=0.5,
        ...     percentile_low=0.5,
        ...     percentile_high=0.5,
        ...     max_dimension=1,
        ...     x_pixel_size_um=0.5,
        ...     scale_bar_um=0.5,
        ... )
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    downsampled = [
        _downsample_cyx_for_qc(image, max_dimension)
        for image in (raw_target_cyx, mip_before_cyx, mip_output_cyx)
    ]
    display_images = [item[0] for item in downsampled]
    display_strides = [item[1] for item in downsampled]
    raw_scale = _raw_scale_rgb(
        display_images,
        rgb_source_channels,
        percentile_high,
    )
    normalized = [
        _independently_normalized_rgb(
            image,
            rgb_source_channels,
            percentile_low,
            percentile_high,
        )
        for image in display_images
    ]

    is_masked = mode in ("raw_masked", "deconv_masked")
    if is_masked:
        retained_percent = 100.0 - float(outside_suppression_percent)
        output_title = (
            "Masked MIP (outside=0)"
            if np.isclose(outside_suppression_percent, 100.0)
            else (
                f"Suppressed MIP (outside retains {retained_percent:g}%)"
            )
        )
        column_titles = (
            f"Raw target Z{target_slice_1based}",
            "MIP before mask",
            output_title,
        )
        selected_raw_scale = raw_scale
        selected_normalized = normalized
        selected_display_images = display_images
        selected_display_strides = display_strides
    else:
        column_titles = (
            f"Raw target Z{target_slice_1based}",
            "Deconvolved MIP" if mode == "deconv_unmasked" else "Raw MIP",
        )
        selected_raw_scale = raw_scale[:2]
        selected_normalized = normalized[:2]
        selected_display_images = display_images[:2]
        selected_display_strides = display_strides[:2]

    column_count = len(column_titles)
    figure, axes = plt.subplots(
        2, column_count, figsize=(5 * column_count, 10), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(2, column_count)
    for column, title in enumerate(column_titles):
        axes[0, column].imshow(selected_raw_scale[column])
        axes[0, column].set_title(f"{title}\nshared intensity display scale")
        axes[1, column].imshow(selected_normalized[column])
        axes[1, column].set_title(
            f"{title}\nindependent P{percentile_low:g}–P{percentile_high:g}"
        )
        for row in range(2):
            _add_scale_bar(
                axes[row, column],
                image_shape_yx=selected_display_images[column].shape[-2:],
                x_pixel_size_um=x_pixel_size_um,
                display_stride=selected_display_strides[column],
                length_um=scale_bar_um,
            )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{sample} | {mode} | merged RGB | {scale_bar_um:g} µm scale bars"
    )
    figure.savefig(output_png, dpi=170)
    plt.close(figure)
    return output_png


def _merged_rss_intensity(cyx: np.ndarray) -> np.ndarray:
    """Return a quantitative merged scalar image without display normalization.

    The merged intensity is the root-sum-square across channels:
    ``sqrt(sum_c I_c**2)``. This avoids arbitrary RGB luminance weights and
    retains the original numerical intensity scale, although channels with
    larger amplitudes contribute more strongly to the merged metric.

    Args:
        cyx (np.ndarray): Array containing cyx.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _merged_rss_intensity(cyx=image_array)
    """
    values = np.asarray(cyx, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(f"Merged SNR expects CYX data, got {values.shape}")
    return np.sqrt(np.sum(np.square(values), axis=0, dtype=np.float64))


def _calculate_mip_snr_rows(
    *,
    sample: str,
    target_slice_1based: int,
    raw_target_cyx: np.ndarray,
    mip_before_cyx: np.ndarray,
    mip_output_cyx: np.ndarray,
    mask_yx: np.ndarray,
    channel_names: Sequence[str],
    epsilon: float,
) -> list[MIPSNRRow]:
    """Compare raw-target and MIP ROI SNR with the current thesis formula.

    Args:
        sample (str): Text value specifying sample.
        target_slice_1based (int): Numerical value controlling target slice 1based.
        raw_target_cyx (np.ndarray): Array containing raw target cyx.
        mip_before_cyx (np.ndarray): Array containing maximum-intensity projection before cyx.
        mip_output_cyx (np.ndarray): Array containing maximum-intensity projection output cyx.
        mask_yx (np.ndarray): Array containing mask yx.
        channel_names (Sequence[str]): Text value specifying channel names.
        epsilon (float): Numerical value controlling epsilon.

    Returns:
        list[MIPSNRRow]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _calculate_mip_snr_rows(
        ...     sample="sample",
        ...     target_slice_1based=1,
        ...     raw_target_cyx=image_array,
        ...     mip_before_cyx=image_array,
        ...     mip_output_cyx=image_array,
        ...     mask_yx=image_array,
        ...     channel_names="channel_names",
        ...     epsilon=0.5,
        ... )
    """
    if raw_target_cyx.shape != mip_before_cyx.shape or mip_before_cyx.shape != mip_output_cyx.shape:
        raise ValueError(
            f"SNR arrays must have identical CYX geometry: raw={raw_target_cyx.shape}, "
            f"before={mip_before_cyx.shape}, after={mip_output_cyx.shape}"
        )
    if len(channel_names) != int(raw_target_cyx.shape[0]):
        raise ValueError(
            f"Expected {raw_target_cyx.shape[0]} channel names, received {len(channel_names)}"
        )

    rows: list[MIPSNRRow] = []
    for channel_index, channel_name in enumerate(channel_names):
        raw_stats = _roi_snr_components(raw_target_cyx[channel_index], mask_yx, epsilon)
        before_stats = _roi_snr_components(mip_before_cyx[channel_index], mask_yx, epsilon)
        after_stats = _roi_snr_components(mip_output_cyx[channel_index], mask_yx, epsilon)
        after_sd = float(after_stats[4])
        if after_sd <= epsilon:
            after_snr: float | None = None
            after_status = "undefined_after_exact_zero_background"
        else:
            after_snr = float(after_stats[-1])
            after_status = "defined"
        rows.append(
            MIPSNRRow(
                sample=sample,
                target_slice_1based=int(target_slice_1based),
                channel_index=int(channel_index),
                channel_name=str(channel_name),
                signal_pixels=int(raw_stats[0]),
                background_pixels=int(raw_stats[1]),
                raw_target_signal_mean=float(raw_stats[2]),
                raw_target_background_mean=float(raw_stats[3]),
                raw_target_background_sd=float(raw_stats[4]),
                raw_target_snr=float(raw_stats[-1]),
                mip_before_signal_mean=float(before_stats[2]),
                mip_before_background_mean=float(before_stats[3]),
                mip_before_background_sd=float(before_stats[4]),
                mip_before_snr=float(before_stats[-1]),
                delta_mip_before_vs_raw_target=float(before_stats[-1] - raw_stats[-1]),
                masked_mip_signal_mean=float(after_stats[2]),
                masked_mip_background_mean=float(after_stats[3]),
                masked_mip_background_sd=after_sd,
                masked_mip_snr=after_snr,
                masked_mip_snr_status=after_status,
            )
        )

    # Add one quantitative merged row. This is not the independently
    # normalized RGB rendering used in the lower QC row. It is computed from
    # original-scale channel values as sqrt(sum(channel**2)).
    raw_merged = _merged_rss_intensity(raw_target_cyx)
    before_merged = _merged_rss_intensity(mip_before_cyx)
    after_merged = _merged_rss_intensity(mip_output_cyx)
    raw_stats = _roi_snr_components(raw_merged, mask_yx, epsilon)
    before_stats = _roi_snr_components(before_merged, mask_yx, epsilon)
    after_stats = _roi_snr_components(after_merged, mask_yx, epsilon)
    after_sd = float(after_stats[4])
    if after_sd <= epsilon:
        after_snr = None
        after_status = "undefined_after_exact_zero_background"
    else:
        after_snr = float(after_stats[-1])
        after_status = "defined"
    rows.append(
        MIPSNRRow(
            sample=sample,
            target_slice_1based=int(target_slice_1based),
            channel_index=-1,
            channel_name="MERGED_RSS_ALL_CHANNELS",
            signal_pixels=int(raw_stats[0]),
            background_pixels=int(raw_stats[1]),
            raw_target_signal_mean=float(raw_stats[2]),
            raw_target_background_mean=float(raw_stats[3]),
            raw_target_background_sd=float(raw_stats[4]),
            raw_target_snr=float(raw_stats[-1]),
            mip_before_signal_mean=float(before_stats[2]),
            mip_before_background_mean=float(before_stats[3]),
            mip_before_background_sd=float(before_stats[4]),
            mip_before_snr=float(before_stats[-1]),
            delta_mip_before_vs_raw_target=float(before_stats[-1] - raw_stats[-1]),
            masked_mip_signal_mean=float(after_stats[2]),
            masked_mip_background_mean=float(after_stats[3]),
            masked_mip_background_sd=after_sd,
            masked_mip_snr=after_snr,
            masked_mip_snr_status=after_status,
        )
    )
    return rows


def _save_mip_snr(
    rows: Sequence[MIPSNRRow],
    *,
    output_dir: Path,
    formula: str,
    outside_suppression_percent: float,
) -> tuple[Path, Path]:
    """Save per-channel plus merged-RSS SNR as compact CSV and JSON files.

    Args:
        rows (Sequence[MIPSNRRow]): Value specifying rows for the operation.
        output_dir (Path): Directory where generated resources are written.
        formula (str): Text value specifying formula.
        outside_suppression_percent (float): Numerical value controlling outside suppression percent.

    Returns:
        tuple[Path, Path]: Resolved or generated filesystem path.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _save_mip_snr(
        ...     rows=[],
        ...     output_dir=Path("path/to/resource"),
        ...     formula="formula",
        ...     outside_suppression_percent=0.5,
        ... )
    """
    import csv

    if not rows:
        raise ValueError("At least one SNR row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "snr_raw_target_vs_mip.csv"
    json_path = output_dir / "snr_raw_target_vs_mip.json"
    dictionaries = [asdict(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0].keys()))
        writer.writeheader()
        writer.writerows(dictionaries)
    channel_rows = [row for row in rows if row.channel_index >= 0]
    merged_rows = [row for row in rows if row.channel_index == -1]
    if len(merged_rows) != 1:
        raise ValueError(
            f"Expected exactly one merged SNR row, received {len(merged_rows)}"
        )
    merged_row = merged_rows[0]
    masked_values = [
        row.masked_mip_snr
        for row in channel_rows
        if row.masked_mip_snr is not None
    ]
    if np.isclose(outside_suppression_percent, 100.0):
        note = (
            "Masked-MIP SNR is undefined after exact zeroing when background standard "
            "deviation is zero; raw-target and pre-mask MIP SNR remain directly comparable."
        )
    else:
        note = (
            f"Outside-mask intensities were suppressed by "
            f"{outside_suppression_percent:g}%. Masked-MIP SNR is mathematically defined "
            "but is inflated by this deliberate background attenuation; use raw-target "
            "and pre-mask MIP SNR for the primary quantitative comparison."
        )
    payload = {
        "formula": formula,
        "roi_mask": "existing_2.5d_unet_prediction",
        "outside_suppression_percent": float(outside_suppression_percent),
        "outside_retained_percent": float(100.0 - outside_suppression_percent),
        "note": note,
        "merged_intensity_definition": "sqrt(sum_over_channels(original_scale_intensity**2))",
        "merged_metric_note": (
            "The merged SNR is quantitative and non-normalized, but channels with "
            "larger numerical amplitudes contribute more strongly. It is not the "
            "independently normalized RGB display image."
        ),
        "rows": dictionaries,
        "mean_raw_target_snr": float(
            np.mean([row.raw_target_snr for row in channel_rows])
        ),
        "mean_mip_before_snr": float(
            np.mean([row.mip_before_snr for row in channel_rows])
        ),
        "mean_delta_mip_before_vs_raw_target": float(
            np.mean([row.delta_mip_before_vs_raw_target for row in channel_rows])
        ),
        "mean_masked_mip_snr": (
            float(np.mean(masked_values)) if masked_values else None
        ),
        "merged_raw_target_snr": float(merged_row.raw_target_snr),
        "merged_mip_before_snr": float(merged_row.mip_before_snr),
        "merged_delta_mip_before_vs_raw_target": float(
            merged_row.delta_mip_before_vs_raw_target
        ),
        "merged_masked_mip_snr": (
            float(merged_row.masked_mip_snr)
            if merged_row.masked_mip_snr is not None
            else None
        ),
        "merged_masked_mip_snr_status": merged_row.masked_mip_snr_status,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def _safe_percent_tag(value: float) -> str:
    """Return safe percent tag for the supplied inputs.

    Args:
        value (float): Value to validate, transform, store, or forward.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _safe_percent_tag(value=0.5)
    """
    rounded = f"{float(value):g}"
    return rounded.replace("-", "m").replace(".", "p")


def mode_output_directory_name(cfg: MIP3DConfig) -> str:
    """Return a collision-safe output directory for the selected product.

    Args:
        cfg (MIP3DConfig): Value specifying cfg for the operation.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = mode_output_directory_name(cfg=config)
    """
    mode = _require_mode(cfg.mode)
    if mode in ("raw_unmasked", "raw_masked"):
        if mode == "raw_unmasked":
            return "raw_mip"
        if np.isclose(cfg.outside_suppression_percent, 100.0):
            return "raw_mip_masked"
        return (
            "raw_mip_suppressed_"
            f"{_safe_percent_tag(cfg.outside_suppression_percent)}pct"
        )

    iteration_short = (
        f"{int(cfg.deconv_iters_blue)}"
        f"{int(cfg.deconv_iters_green)}"
        f"{int(cfg.deconv_iters_red)}"
    )
    if mode == "deconv_unmasked":
        return f"deconv_{iteration_short}_mip"
    if np.isclose(cfg.outside_suppression_percent, 100.0):
        return f"deconv_{iteration_short}_mip_masked"
    return (
        f"deconv_{iteration_short}_mip_suppressed_"
        f"{_safe_percent_tag(cfg.outside_suppression_percent)}pct"
    )


def _output_sample_dir(cfg: MIP3DConfig, raw_zarr: Path) -> Path:
    """Return output sample dir for the supplied inputs.

    Args:
        cfg (MIP3DConfig): Value specifying cfg for the operation.
        raw_zarr (Path): Filesystem path used for raw Zarr.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _output_sample_dir(cfg=config, raw_zarr=Path("path/to/resource"))
    """
    project_root = Path(cfg.project_root or find_project_root(Path(__file__).resolve())).resolve()
    output_root = resolve_project_path(
        cfg.output_root or Path("results") / "mip_2d",
        project_root,
    )
    return output_root / mode_output_directory_name(cfg) / relative_volume_path(raw_zarr)


def create_mip_for_volume(cfg: MIP3DConfig) -> MIP3DOutput:
    """Create and save one selected 2D MIP product for one configured raw volume.

    Args:
        cfg (MIP3DConfig): Value specifying cfg for the operation.

    Returns:
        MIP3DOutput: Result produced by the operation.

    Raises:
        FileExistsError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = create_mip_for_volume(cfg=config)
    """
    if cfg.raw_zarr is None:
        raise ValueError("raw_zarr is required")
    if int(cfg.level) != 0:
        raise ValueError("The current 3D MIP workflow is fixed to source level 0")
    mode = _require_mode(cfg.mode)
    _validate_iteration_count(cfg.deconv_iters_blue, "blue iterations")
    _validate_iteration_count(cfg.deconv_iters_green, "green iterations")
    _validate_iteration_count(cfg.deconv_iters_red, "red iterations")
    suppression = float(cfg.outside_suppression_percent)
    if not np.isfinite(suppression) or not 0.0 <= suppression <= 100.0:
        raise ValueError(
            "outside_suppression_percent must be between 0 and 100, "
            f"got {cfg.outside_suppression_percent}"
        )

    project_root = Path(
        cfg.project_root or find_project_root(Path(__file__).resolve())
    ).expanduser().resolve()
    raw_zarr = resolve_project_path(cfg.raw_zarr, project_root)
    raw_root = resolve_project_path(
        cfg.raw_root or Path("results") / "img" / "3d_data",
        project_root,
    )
    deconv_root = resolve_project_path(
        cfg.deconv_root or Path("results") / "deconv",
        project_root,
    )
    mask_root = resolve_project_path(
        cfg.mask_root or Path("results") / "U-net" / "3d_25d",
        project_root,
    )

    # Validate that the raw source belongs to the configured mapping and read the
    # raw target before selecting a possibly deconvolved projection source.
    target_slice = target_slice_for_volume(raw_zarr, image_root=raw_root)
    sample = volume_key(raw_zarr, image_root=raw_root)
    raw_array, raw_axes, _raw_metadata = _open_level_array(raw_zarr, cfg.level)
    raw_array, raw_axes = _canonicalize_czyx_view(raw_array, raw_axes, raw_zarr)
    _require_czyx(raw_array, raw_axes, raw_zarr, cfg.expected_z_count)
    raw_target_cyx = _read_cyx_slice(raw_array, target_slice)

    projection_source_zarr = (
        resolve_deconvolved_zarr(
            raw_zarr,
            deconv_root,
            iters_blue=cfg.deconv_iters_blue,
            iters_green=cfg.deconv_iters_green,
            iters_red=cfg.deconv_iters_red,
        )
        if mode.startswith("deconv_")
        else raw_zarr
    )
    source_array, source_axes, _source_metadata = _open_level_array(
        projection_source_zarr, cfg.level
    )
    source_array, source_axes = _canonicalize_czyx_view(
        source_array, source_axes, projection_source_zarr
    )
    _require_czyx(
        source_array,
        source_axes,
        projection_source_zarr,
        cfg.expected_z_count,
    )
    mip_before = maximum_intensity_projection_cyx(source_array)

    if mode.startswith("deconv_"):
        mip_before = mip_before.astype(np.float32, copy=False)
        if mip_before.dtype != np.float32:
            raise ValueError("Deconvolved MIP must be float32")

    mask_zarr: Path | None = None
    mask_yx: np.ndarray | None = None
    if mode in ("raw_masked", "deconv_masked"):
        mask_zarr = resolve_predicted_mask_zarr(raw_zarr, mask_root)
        mask_yx = _load_target_mask_yx(
            mask_zarr,
            target_slice_1based=target_slice,
            expected_yx=tuple(int(value) for value in mip_before.shape[1:]),
        )
        mip_output = apply_background_suppression_mask(
            mip_before,
            mask_yx,
            suppression_percent=cfg.outside_suppression_percent,
        )
    else:
        mip_output = mip_before.copy()

    if mode.startswith("raw_") and mip_output.dtype != np.dtype(raw_array.dtype):
        raise ValueError(
            f"Raw MIP dtype changed unexpectedly: {mip_output.dtype} != {raw_array.dtype}"
        )
    if mode.startswith("deconv_") and mip_output.dtype != np.float32:
        raise ValueError(f"Deconvolved MIP must be float32, got {mip_output.dtype}")

    optics = resolve_channel_optics(raw_zarr, level=cfg.level)
    channel_names = [item.name for item in optics]
    snr_rows: list[MIPSNRRow] = []
    if mask_yx is not None:
        snr_rows = _calculate_mip_snr_rows(
            sample=sample,
            target_slice_1based=target_slice,
            raw_target_cyx=raw_target_cyx,
            mip_before_cyx=mip_before,
            mip_output_cyx=mip_output,
            mask_yx=mask_yx,
            channel_names=channel_names,
            epsilon=cfg.snr_epsilon,
        )

    sample_dir = _output_sample_dir(cfg, raw_zarr)
    if sample_dir.exists() and cfg.overwrite:
        shutil.rmtree(sample_dir)
    elif sample_dir.exists():
        raise FileExistsError(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cyx_scale = _cyx_coordinate_scale(projection_source_zarr, level=cfg.level)
    portable_unmasked_mip_zarr: Path | None = None
    portable_mask_zarr: Path | None = None
    if mask_yx is not None:
        portable_unmasked_mip_zarr = save_ome_zarr(
            sample_dir / "unmasked_mip.ome.zarr",
            mip_before,
            "cyx",
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
            coordinate_scale=cyx_scale,
            extra_attrs={
                "pft_processing": {
                    "operation": "portable_unmasked_mip_for_segmentation_preparation",
                    "source_omezarr": str(projection_source_zarr),
                    "source_project_relative": project_relative_path(
                        projection_source_zarr, project_root
                    ),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        mask_scale = cyx_scale[-2:] if cyx_scale and len(cyx_scale) >= 2 else None
        portable_mask_zarr = save_ome_zarr(
            sample_dir / "foreground_mask.ome.zarr",
            mask_yx.astype(np.uint8, copy=False),
            "yx",
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
            coordinate_scale=mask_scale,
            extra_attrs={
                "pft_processing": {
                    "operation": "portable_2d_foreground_mask_for_mip",
                    "source_mask_omezarr": str(mask_zarr),
                    "source_mask_project_relative": project_relative_path(
                        mask_zarr, project_root
                    ) if mask_zarr is not None else None,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    retained = copyable_root_metadata(projection_source_zarr)
    processing = {
        "operation": "maximum_intensity_projection_3d_to_2d",
        "mode": mode,
        "raw_source_omezarr": str(raw_zarr),
        "raw_source_project_relative": project_relative_path(raw_zarr, project_root),
        "projection_source_omezarr": str(projection_source_zarr),
        "projection_source_project_relative": project_relative_path(
            projection_source_zarr, project_root
        ),
        "portable_unmasked_mip_relative": (
            portable_unmasked_mip_zarr.name
            if portable_unmasked_mip_zarr is not None
            else None
        ),
        "source_level": int(cfg.level),
        "projected_axis": "z",
        "projected_z_count": int(source_array.shape[1]),
        "projected_z_range_1based": [1, int(source_array.shape[1])],
        "target_slice_1based_for_qc_and_mask": int(target_slice),
        "mask_omezarr": str(mask_zarr) if mask_zarr is not None else None,
        "mask_project_relative": (
            project_relative_path(mask_zarr, project_root)
            if mask_zarr is not None
            else None
        ),
        "portable_mask_relative": (
            portable_mask_zarr.name if portable_mask_zarr is not None else None
        ),
        "mask_source": "existing_2.5d_unet_target_prediction" if mask_zarr else None,
        "mask_applied_after_projection": bool(mask_zarr),
        "outside_mask_suppression_percent": (
            float(cfg.outside_suppression_percent) if mask_zarr else None
        ),
        "outside_mask_retained_fraction": (
            float(1.0 - cfg.outside_suppression_percent / 100.0)
            if mask_zarr
            else None
        ),
        "outside_mask_operation": (
            "multiply_by_retained_fraction" if mask_zarr else None
        ),
        "deconvolution_model": DECONV_MODEL if mode.startswith("deconv_") else None,
        "deconvolution_iterations": (
            {
                "blue": int(cfg.deconv_iters_blue),
                "green": int(cfg.deconv_iters_green),
                "red": int(cfg.deconv_iters_red),
            }
            if mode.startswith("deconv_")
            else None
        ),
        "stored_normalization": "none",
        "stored_dtype_policy": (
            "float32_from_deconvolution"
            if mode.startswith("deconv_")
            else "preserve_raw_source_dtype"
        ),
        "display_qc": "merged_rgb_only; shared raw scale and independent percentile normalization",
        "qc_scale_bar_um": float(cfg.scale_bar_um),
        "snr_formula": "(foreground_mean - background_mean) / (background_sample_sd + epsilon)",
        "snr_roi_mask": "existing_2.5d_unet_prediction" if mask_zarr else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_attrs = dict(retained)
    output_attrs["pft_processing"] = processing
    output_zarr = save_ome_zarr(
        sample_dir / "image.ome.zarr",
        mip_output,
        "cyx",
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        coordinate_scale=cyx_scale,
        extra_attrs=output_attrs,
    )

    rgb_source_channels = resolve_rgb_source_channels(raw_zarr, level=cfg.level)
    qc_png = _save_merged_qc(
        raw_target_cyx=raw_target_cyx,
        mip_before_cyx=mip_before,
        mip_output_cyx=mip_output,
        rgb_source_channels=rgb_source_channels,
        target_slice_1based=target_slice,
        sample=sample,
        mode=mode,
        output_png=sample_dir / "qc" / "merged_mip_vs_raw_target.png",
        percentile_low=cfg.qc_percentile_low,
        percentile_high=cfg.qc_percentile_high,
        max_dimension=cfg.qc_max_dimension,
        x_pixel_size_um=_x_pixel_size_um(raw_zarr, level=cfg.level),
        scale_bar_um=cfg.scale_bar_um,
        outside_suppression_percent=cfg.outside_suppression_percent,
    )

    snr_csv: Path | None = None
    snr_json: Path | None = None
    if snr_rows:
        snr_csv, snr_json = _save_mip_snr(
            snr_rows,
            output_dir=sample_dir,
            formula="(foreground_mean - background_mean) / (background_sample_sd + epsilon)",
            outside_suppression_percent=cfg.outside_suppression_percent,
        )

    report = {
        "sample": sample,
        "mode": mode,
        "raw_zarr": str(raw_zarr),
        "projection_source_zarr": str(projection_source_zarr),
        "mask_zarr": str(mask_zarr) if mask_zarr else None,
        "portable_unmasked_mip_zarr": (
            str(portable_unmasked_mip_zarr) if portable_unmasked_mip_zarr else None
        ),
        "portable_mask_zarr": str(portable_mask_zarr) if portable_mask_zarr else None,
        "output_zarr": str(output_zarr),
        "qc_png": str(qc_png),
        "scale_bar_um": float(cfg.scale_bar_um),
        "snr_csv": str(snr_csv) if snr_csv else None,
        "snr_json": str(snr_json) if snr_json else None,
        "snr_rows": [asdict(row) for row in snr_rows],
        "target_slice_1based": int(target_slice),
        "source_shape_czyx": [int(value) for value in source_array.shape],
        "source_dtype": str(np.dtype(source_array.dtype)),
        "mip_before_shape_cyx": list(mip_before.shape),
        "mip_before_dtype": str(mip_before.dtype),
        "output_shape_cyx": list(mip_output.shape),
        "output_dtype": str(mip_output.dtype),
        "output_min": float(np.min(mip_output)),
        "output_max": float(np.max(mip_output)),
        "foreground_pixels": int(np.count_nonzero(mask_yx)) if mask_yx is not None else None,
        "background_pixels": int(mask_yx.size - np.count_nonzero(mask_yx)) if mask_yx is not None else None,
        "configuration": asdict(cfg),
        "processing": processing,
    }
    report_json = sample_dir / "mip_3d_to_2d_report.json"
    report_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return MIP3DOutput(
        sample=sample,
        mode=mode,
        raw_zarr=raw_zarr,
        projection_source_zarr=projection_source_zarr,
        mask_zarr=mask_zarr,
        portable_unmasked_mip_zarr=portable_unmasked_mip_zarr,
        portable_mask_zarr=portable_mask_zarr,
        output_zarr=output_zarr,
        qc_png=qc_png,
        report_json=report_json,
        snr_csv=snr_csv,
        snr_json=snr_json,
        target_slice_1based=int(target_slice),
        output_dtype=str(mip_output.dtype),
        output_shape_cyx=tuple(int(value) for value in mip_output.shape),
    )


__all__ = [
    "DECONV_DEFAULT_ITERS_BLUE",
    "DECONV_DEFAULT_ITERS_GREEN",
    "DECONV_DEFAULT_ITERS_RED",
    "DECONV_MODEL",
    "EXPECTED_Z_COUNT",
    "MIP3DConfig",
    "MIP3DOutput",
    "MIPSNRRow",
    "MIPMode",
    "MODE_DIRECTORY_NAMES",
    "apply_background_suppression_mask",
    "apply_zero_background_mask",
    "create_mip_for_volume",
    "deconvolution_iteration_tag",
    "maximum_intensity_projection_cyx",
    "mode_output_directory_name",
    "resolve_deconvolved_332_zarr",
    "resolve_deconvolved_zarr",
    "resolve_predicted_mask_zarr",
]
