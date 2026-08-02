"""Provide command-line and programmatic utilities for denoise widget."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QMessageBox,
    QCheckBox,
)

try:
    from napari.qt.threading import thread_worker
except Exception:  # pragma: no cover
    thread_worker = None

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr


@dataclass
class StepLayerResult:
    """Store validated configuration or result data for step layer result."""
    data: np.ndarray
    name: str
    layer_type: str = "image"
    kwargs: dict[str, Any] | None = None


@dataclass
class DenoiseRequest:
    """Store validated configuration or result data for denoise request."""
    mode: str
    dataset: str
    project_root: Path
    active_layer_name: str
    active_channel_index: int | None
    active_channel_name: str | None
    active_axes: str
    active_scale: tuple[float, ...] | None
    active_colormap: str | None
    source_ome_zarr: str | None
    layers: list[tuple[str, np.ndarray, dict[str, Any]]]
    filter_high_percentile: float
    unet_threshold: float
    unet_patch: int
    model_path: Path | None
    psf_model: str
    rl_iterations: int
    rl_background: float
    psf_path: Path | None
    add_mask_layer: bool
    add_refined_layer: bool


def make_denoise_widget(napari_viewer):
    """Create denoise widget from the supplied inputs.

    Args:
        napari_viewer (Any): Value specifying napari viewer for the operation.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = make_denoise_widget(napari_viewer=...)
    """
    return PFTDenoiseWidget(napari_viewer)


def _safe_name(text: str) -> str:
    """Return safe name for the supplied inputs.

    Args:
        text (str): Text value specifying text.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _safe_name(text="text")
    """
    bad = r'<>:"/\\|?*• '
    cleaned = "".join("_" if ch in bad else ch for ch in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "layer"


def _as_numpy(data: Any) -> np.ndarray:
    """Return as numpy for the supplied inputs.

    Args:
        data (Any): Value specifying data for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _as_numpy(data=image_array)
    """
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def _layer_axes(layer: Any) -> str:
    """Return layer axes for the supplied inputs.

    Args:
        layer (Any): Image, labels, shapes, or points layer used by the graphical operation.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _layer_axes(layer=...)
    """
    md = getattr(layer, "metadata", {}) or {}
    axes = md.get("pft_layer_axes") or md.get("pft_axes")
    if isinstance(axes, str) and axes:
        return axes.lower()
    ndim = int(np.ndim(layer.data))
    if ndim == 2:
        return "yx"
    if ndim == 3:
        return "zyx"
    if ndim == 4:
        return "czyx"
    return "".join(f"d{i}" for i in range(ndim))


def _layer_scale(layer: Any) -> tuple[float, ...] | None:
    """Return layer scale for the supplied inputs.

    Args:
        layer (Any): Image, labels, shapes, or points layer used by the graphical operation.

    Returns:
        tuple[float, ...] | None: Collection containing the generated or selected values.

    Example:
        >>> result = _layer_scale(layer=...)
    """
    scale = getattr(layer, "scale", None)
    if scale is None:
        return None
    try:
        return tuple(float(v) for v in scale)
    except Exception:
        return None


def _contrast_limits(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> tuple[float, float]:
    """Return contrast limits for the supplied inputs.

    Args:
        data (np.ndarray): Array containing data.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        tuple[float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = _contrast_limits(data=image_array)
    """
    x = np.asarray(data)
    if x.size == 0:
        return 0.0, 1.0
    flat = x.reshape(-1)
    max_points = 2_000_000
    if flat.size > max_points:
        flat = flat[:: max(1, flat.size // max_points)]
    lo, hi = np.percentile(flat, [p_low, p_high])
    lo = float(lo)
    hi = float(hi if hi > lo else lo + 1.0)
    return lo, hi


def _robust_normalize(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> tuple[np.ndarray, float, float]:
    """Return robust normalize for the supplied inputs.

    Args:
        data (np.ndarray): Array containing data.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        tuple[np.ndarray, float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = _robust_normalize(data=image_array)
    """
    x = np.asarray(data, dtype=np.float32)
    lo, hi = _contrast_limits(x, p_low=p_low, p_high=p_high)
    y = (x - np.float32(lo)) / np.float32(hi - lo + 1e-8)
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32), float(lo), float(hi)


def _denormalize(data_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return denormalize for the supplied inputs.

    Args:
        data_norm (np.ndarray): Array containing data norm.
        lo (float): Numerical value controlling lo.
        hi (float): Numerical value controlling hi.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _denormalize(
        ...     data_norm=image_array,
        ...     lo=0.5,
        ...     hi=0.5,
        ... )
    """
    y = np.asarray(data_norm, dtype=np.float32)
    out = y * np.float32(hi - lo) + np.float32(lo)
    out = np.clip(out, np.float32(lo), np.float32(hi))
    return out.astype(np.float32)


def _default_model_path(project_root: Path, dataset: str) -> Path:
    """Return default model path for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _default_model_path(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    if dataset == "3d_25d":
        return project_root / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras"
    return project_root / "models" / f"u_net_{dataset}" / f"u_net_{dataset}_best.keras"


def _infer_dataset_from_layers(active_layer: Any, sibling_layers: list[Any]) -> str:
    """Infer dataset from layers from the supplied model inputs.

    Args:
        active_layer (Any): Value specifying active layer for the operation.
        sibling_layers (list[Any]): Value specifying sibling layers for the operation.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _infer_dataset_from_layers(active_layer=..., sibling_layers=[])
    """
    axes = _layer_axes(active_layer)
    if "z" in axes:
        return "3d_25d"
    if len(sibling_layers) >= 2:
        return "2d_wga_dapi"
    return "2d_time"


def _is_image_layer(layer: Any) -> bool:
    """Determine whether image layer satisfies the stated condition.

    Args:
        layer (Any): Image, labels, shapes, or points layer used by the graphical operation.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _is_image_layer(layer=...)
    """
    return hasattr(layer, "data") and hasattr(layer, "metadata") and layer.__class__.__name__.lower() != "labels"


def _is_pft_layer(layer: Any) -> bool:
    """Determine whether pft layer satisfies the stated condition.

    Args:
        layer (Any): Image, labels, shapes, or points layer used by the graphical operation.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _is_pft_layer(layer=...)
    """
    name = str(getattr(layer, "name", ""))
    md = getattr(layer, "metadata", {}) or {}
    return name.startswith("PFT |") or any(str(k).startswith("pft_") for k in md)


def _same_source(a: Any, b: Any) -> bool:
    """Return same source for the supplied inputs.

    Args:
        a (Any): Value specifying a for the operation.
        b (Any): Value specifying b for the operation.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _same_source(a=..., b=...)
    """
    ma = getattr(a, "metadata", {}) or {}
    mb = getattr(b, "metadata", {}) or {}
    za = ma.get("pft_ome_zarr_path") or ma.get("pft_source_path")
    zb = mb.get("pft_ome_zarr_path") or mb.get("pft_source_path")
    return bool(za and zb and str(za) == str(zb))


def _channel_index(layer: Any) -> int:
    """Return channel index for the supplied inputs.

    Args:
        layer (Any): Image, labels, shapes, or points layer used by the graphical operation.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _channel_index(layer=...)
    """
    md = getattr(layer, "metadata", {}) or {}
    try:
        return int(md.get("pft_channel_index", 0))
    except Exception:
        return 0


def _channel_name(layer: Any) -> str:
    """Return channel name for the supplied inputs.

    Args:
        layer (Any): Image, labels, shapes, or points layer used by the graphical operation.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _channel_name(layer=...)
    """
    md = getattr(layer, "metadata", {}) or {}
    return str(md.get("pft_channel_name") or getattr(layer, "name", "channel"))


def _guess_wavelength_nm(channel_name: str | None, channel_index: int | None) -> int:
    """Return guess wavelength nm for the supplied inputs.

    Args:
        channel_name (str | None): Text value specifying channel name.
        channel_index (int | None): Zero-based index selecting channel.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _guess_wavelength_nm(channel_name="channel_name", channel_index=1)
    """
    text = (channel_name or "").lower()
    if any(k in text for k in ("405", "dapi", "hada", "t1", "blue")):
        return 405
    if any(k in text for k in ("488", "wga", "nada", "t2", "green")):
        return 488
    if any(k in text for k in ("561", "tada", "t3", "red")):
        return 561
    return {0: 405, 1: 488, 2: 561}.get(int(channel_index or 0), 488)


def _find_psf_path(project_root: Path, *, psf_model: str, channel_name: str | None, channel_index: int | None, level: int = 0) -> Path | None:
    """Find point-spread function path in the available data or project structure.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        psf_model (str): Text value specifying point-spread function model.
        channel_name (str | None): Text value specifying channel name.
        channel_index (int | None): Zero-based index selecting channel.
        level (int): Numerical value controlling level. Defaults to ``0``.

    Returns:
        Path | None: Resolved or generated filesystem path.

    Example:
        >>> result = _find_psf_path(
        ...     project_root=Path("path/to/resource"),
        ...     psf_model="psf_model",
        ...     channel_name="channel_name",
        ...     channel_index=1,
        ... )
    """
    psf_dir = project_root / "results" / "psf" / "generated"
    if not psf_dir.exists():
        return None
    lam = _guess_wavelength_nm(channel_name, channel_index)
    channel_text = _safe_name(channel_name or "")
    candidates: list[Path] = []
    if channel_text:
        candidates.append(psf_dir / f"psf_{psf_model}_{channel_text}_Lambda{lam}nm__L{level}.tif")
    candidates.extend(sorted(psf_dir.glob(f"psf_{psf_model}_*Lambda{lam}nm__L{level}.tif")))
    candidates.extend(sorted(psf_dir.glob(f"psf_{psf_model}_*__L{level}.tif")))
    for path in candidates:
        if path.exists():
            return path
    return None


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Normalize point-spread function using the configured procedure.

    Args:
        psf (np.ndarray): Array containing point-spread function.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _normalize_psf(psf=image_array)
    """
    psf = np.asarray(psf, dtype=np.float32)
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D (Z,Y,X). Got shape={psf.shape}")
    if not np.isfinite(psf).all():
        raise ValueError("PSF contains NaN or Inf values.")
    psf = np.maximum(psf, 0.0)
    total = float(psf.sum())
    if total <= 0:
        raise ValueError("PSF sum is zero after clipping negative values.")
    return psf / total


def _richardson_lucy_volume(image_zyx: np.ndarray, psf_path: Path, *, iterations: int, background: float) -> np.ndarray:
    """Return richardson lucy volume for the supplied inputs.

    Args:
        image_zyx (np.ndarray): Array containing image zyx.
        psf_path (Path): Filesystem path associated with point-spread function.
        iterations (int): Numerical value controlling iterations.
        background (float): Numerical value controlling background.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _richardson_lucy_volume(
        ...     image_zyx=image_array,
        ...     psf_path=Path("path/to/resource"),
        ...     iterations=1,
        ...     background=0.5,
        ... )
    """
    import tifffile as tiff
    from skimage.restoration import richardson_lucy

    image = np.asarray(image_zyx, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"3D deconvolution expects ZYX data. Got shape={image.shape}")
    image = np.maximum(image - np.float32(background), 0.0)
    psf = _normalize_psf(tiff.imread(str(psf_path)))
    if any(p > i for p, i in zip(psf.shape, image.shape)):
        raise ValueError(f"PSF shape {psf.shape} is larger than image shape {image.shape}.")
    out = richardson_lucy(image=image, psf=psf, num_iter=int(iterations), clip=False)
    return np.asarray(out, dtype=np.float32)


def _filter_2d_or_time(data: np.ndarray, high_percentile: float) -> np.ndarray:
    """Filter two-dimensional data or time according to the configured criteria.

    Args:
        data (np.ndarray): Array containing data.
        high_percentile (float): Numerical value controlling high percentile.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _filter_2d_or_time(data=image_array, high_percentile=0.5)
    """
    from PFT.core_prog_parts.denoiser import LocalHighThresholdSpec, apply_local_high_threshold_single

    spec = LocalHighThresholdSpec(high_percentile=float(high_percentile))
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 2:
        filtered, _, _ = apply_local_high_threshold_single(x, spec)
        return filtered.astype(np.float32)
    if x.ndim == 3:
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            filtered, _, _ = apply_local_high_threshold_single(x[i], spec)
            out[i] = filtered
        return out
    raise ValueError(f"2D filtering expects YX or TYX. Got shape={x.shape}")


def _run_unet_2d_or_time(data_hwc_or_thwc: np.ndarray, *, model_path: Path, dataset: str, threshold: float, patch: int, project_root: Path) -> np.ndarray:
    """Run U-Net result two-dimensional data or time using the supplied configuration.

    Args:
        data_hwc_or_thwc (np.ndarray): Array containing data hwc or thwc.
        model_path (Path): Filesystem path associated with model.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        threshold (float): Numerical decision threshold used by the operation.
        patch (int): Numerical value controlling patch.
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _run_unet_2d_or_time(
        ...     data_hwc_or_thwc=image_array,
        ...     model_path=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     threshold=0.5,
        ...     patch=1,
        ...     project_root=Path("path/to/resource"),
        ... )
    """
    from PFT.core_prog_parts.unet_run_core import UNetRunConfig, load_unet_model, predict_2d_tiled

    cfg = UNetRunConfig(project_root=project_root, dataset=dataset, threshold=float(threshold), patch=int(patch), model_path=Path(model_path))
    model = load_unet_model(Path(model_path))
    x = np.asarray(data_hwc_or_thwc, dtype=np.float32)
    if x.ndim == 3:
        return predict_2d_tiled(model, x, cfg).astype(np.uint8)
    if x.ndim == 4:
        pred = np.zeros(x.shape[:3], dtype=np.uint8)
        for t in range(x.shape[0]):
            pred[t] = predict_2d_tiled(model, x[t], cfg).astype(np.uint8)
        return pred
    raise ValueError(f"U-Net 2D expects HWC or THWC. Got shape={x.shape}")


def _run_unet_3d_25d(img_czyx: np.ndarray, *, model_path: Path, threshold: float, patch: int, project_root: Path) -> np.ndarray:
    """Run U-Net result three-dimensional data 25d using the supplied configuration.

    Args:
        img_czyx (np.ndarray): Array containing img czyx.
        model_path (Path): Filesystem path associated with model.
        threshold (float): Numerical decision threshold used by the operation.
        patch (int): Numerical value controlling patch.
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _run_unet_3d_25d(
        ...     img_czyx=image_array,
        ...     model_path=Path("path/to/resource"),
        ...     threshold=0.5,
        ...     patch=1,
        ...     project_root=Path("path/to/resource"),
        ... )
    """
    from PFT.core_prog_parts.unet_run_core import UNetRunConfig, load_unet_model, predict_2d_tiled
    from PFT.core_prog_parts.unet_train_3d_25d_core import make_25d_input_slice

    x = np.asarray(img_czyx, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"3D U-Net expects C,Z,Y,X. Got shape={x.shape}")
    cfg = UNetRunConfig(
        project_root=project_root,
        dataset="3d_25d",
        threshold=float(threshold),
        patch=int(patch),
        model_path=Path(model_path),
        channels=tuple(range(x.shape[0])),
    )
    model = load_unet_model(Path(model_path))
    _, z_total, y, x_size = x.shape
    pred = np.zeros((z_total, y, x_size), dtype=np.uint8)
    for z in range(z_total):
        plane_hwc = make_25d_input_slice(x, z, cfg)
        pred[z] = predict_2d_tiled(model, plane_hwc, cfg).astype(np.uint8)
    return pred


def _prepare_2d_stack(layers: list[tuple[str, np.ndarray, dict[str, Any]]], high_percentile: float):
    """Prepare two-dimensional data stack for downstream processing.

    Args:
        layers (list[tuple[str, np.ndarray, dict[str, Any]]]): Array containing layers.
        high_percentile (float): Numerical value controlling high percentile.

    Returns:
        Any: Result produced by the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _prepare_2d_stack(layers=image_array, high_percentile=0.5)
    """
    prepared = []
    filtered_norm_arrays: list[np.ndarray] = []
    for name, arr, md in layers:
        axes = str(md.get("axes") or "").lower()
        if "z" in axes:
            raise ValueError("2D stack preparation received Z data.")
        arr_norm, lo, hi = _robust_normalize(arr)
        filtered_norm = _filter_2d_or_time(arr_norm, high_percentile)
        filtered_raw = _denormalize(filtered_norm, lo, hi)
        prepared.append((name, filtered_raw, filtered_norm, {**md, "pft_norm_lo": lo, "pft_norm_hi": hi}))
        filtered_norm_arrays.append(filtered_norm)
    first = filtered_norm_arrays[0]
    if first.ndim == 2:
        img = np.stack(filtered_norm_arrays, axis=-1)
    elif first.ndim == 3:
        img = np.stack(filtered_norm_arrays, axis=-1)
    else:
        raise ValueError(f"Unexpected filtered array shape: {first.shape}")
    return prepared, img


def _denoise_pipeline_worker(req: DenoiseRequest) -> Iterable[StepLayerResult]:
    """Return denoise pipeline worker for the supplied inputs.

    Args:
        req (DenoiseRequest): Value specifying req for the operation.

    Returns:
        Iterable[StepLayerResult]: Iterator yielding the generated or selected values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _denoise_pipeline_worker(req=...)
    """
    if not req.layers:
        raise ValueError("No image layers were selected for processing.")
    model_path = req.model_path or _default_model_path(req.project_root, req.dataset)
    if not model_path.exists():
        raise FileNotFoundError(f"U-Net model not found: {model_path}")

    if req.mode == "2d":
        prepared, unet_input_norm = _prepare_2d_stack(req.layers, req.filter_high_percentile)
        for name, filtered_raw, _filtered_norm, md in prepared:
            kwargs = {
                "name": f"PFT | denoise | filtered | {name}",
                "metadata": {
                    **md,
                    "pft_step": "local_high_filter",
                    "pft_variant": "raw",
                    "pft_is_normalized": False,
                    "pft_processing_domain": "normalized",
                    "pft_filter_high_percentile": req.filter_high_percentile,
                    "pft_save_default": False,
                },
                "blending": "additive",
            }
            if req.active_scale is not None and len(req.active_scale) == filtered_raw.ndim:
                kwargs["scale"] = req.active_scale
            kwargs["contrast_limits"] = _contrast_limits(filtered_raw)
            yield StepLayerResult(filtered_raw.astype(np.float32), kwargs["name"], "image", kwargs)

        mask = _run_unet_2d_or_time(
            unet_input_norm,
            model_path=model_path,
            dataset=req.dataset,
            threshold=req.unet_threshold,
            patch=req.unet_patch,
            project_root=req.project_root,
        )

        if req.add_mask_layer:
            mask_kwargs = {
                "name": "PFT | denoise | U-Net mask",
                "metadata": {
                    "pft_step": "unet_mask",
                    "pft_dataset": req.dataset,
                    "pft_model_path": str(model_path),
                    "pft_save_default": False,
                    "pft_axes": "tyx" if mask.ndim == 3 else "yx",
                    "pft_ome_zarr_path": req.source_ome_zarr,
                },
            }
            if req.active_scale is not None and len(req.active_scale) == mask.ndim:
                mask_kwargs["scale"] = req.active_scale
            yield StepLayerResult(mask.astype(np.uint8), mask_kwargs["name"], "labels", mask_kwargs)

        if req.add_refined_layer:
            for name, _filtered_raw, filtered_norm, md in prepared:
                refined_norm = filtered_norm.astype(np.float32) * mask.astype(np.float32)
                refined_raw = _denormalize(refined_norm, md["pft_norm_lo"], md["pft_norm_hi"])
                kwargs = {
                    "name": f"PFT | denoise | U-Net refined | {name}",
                    "metadata": {
                        **md,
                        "pft_step": "unet_refined_image",
                        "pft_dataset": req.dataset,
                        "pft_model_path": str(model_path),
                        "pft_variant": "raw",
                        "pft_is_normalized": False,
                        "pft_processing_domain": "normalized",
                        "pft_save_default": False,
                    },
                    "blending": "additive",
                }
                if req.active_scale is not None and len(req.active_scale) == refined_raw.ndim:
                    kwargs["scale"] = req.active_scale
                kwargs["contrast_limits"] = _contrast_limits(refined_raw)
                yield StepLayerResult(refined_raw.astype(np.float32), kwargs["name"], "image", kwargs)
        return

    if req.mode == "3d":
        prepared = []
        for name, arr, md in req.layers:
            axes = str(md.get("axes") or "zyx").lower()
            if arr.ndim != 3 or "z" not in axes:
                raise ValueError(f"3D mode expects ZYX channel layers. Got {name}: shape={arr.shape}, axes={axes}")
            arr_norm, lo, hi = _robust_normalize(arr)
            prepared.append((name, arr.astype(np.float32), arr_norm, {**md, "pft_norm_lo": lo, "pft_norm_hi": hi}))

        img_czyx_norm = np.stack([arr_norm for _, _, arr_norm, _ in prepared], axis=0)
        mask = _run_unet_3d_25d(
            img_czyx_norm,
            model_path=model_path,
            threshold=req.unet_threshold,
            patch=req.unet_patch,
            project_root=req.project_root,
        )

        if req.add_mask_layer:
            mask_kwargs = {
                "name": "PFT | denoise | U-Net mask | 3D",
                "metadata": {
                    "pft_step": "unet_3d_25d_mask",
                    "pft_dataset": "3d_25d",
                    "pft_model_path": str(model_path),
                    "pft_save_default": False,
                    "pft_axes": "zyx",
                    "pft_ome_zarr_path": req.source_ome_zarr,
                },
            }
            if req.active_scale is not None and len(req.active_scale) == 3:
                mask_kwargs["scale"] = req.active_scale
            yield StepLayerResult(mask.astype(np.uint8), mask_kwargs["name"], "labels", mask_kwargs)

        active_index = 0
        for i, (name, _raw, _norm, _md) in enumerate(prepared):
            if name == req.active_layer_name:
                active_index = i
                break

        active_channel_name = req.active_channel_name or prepared[active_index][0]
        psf_path = req.psf_path or _find_psf_path(
            req.project_root,
            psf_model=req.psf_model,
            channel_name=active_channel_name,
            channel_index=req.active_channel_index,
            level=0,
        )
        if psf_path is None or not psf_path.exists():
            raise FileNotFoundError(
                "PSF file was not found. Set PSF path manually or generate PSFs first under "
                f"{req.project_root / 'results' / 'psf' / 'generated'}"
            )

        _name, _raw_active, norm_active, md_active = prepared[active_index]
        deconv_norm = _richardson_lucy_volume(norm_active, psf_path, iterations=req.rl_iterations, background=req.rl_background)
        deconv_raw = _denormalize(deconv_norm, md_active["pft_norm_lo"], md_active["pft_norm_hi"])

        deconv_kwargs = {
            "name": f"PFT | denoise | deconvolved | {active_channel_name}",
            "metadata": {
                **md_active,
                "pft_step": "richardson_lucy_deconvolution",
                "pft_psf_path": str(psf_path),
                "pft_rl_iterations": req.rl_iterations,
                "pft_rl_background": req.rl_background,
                "pft_variant": "raw",
                "pft_is_normalized": False,
                "pft_processing_domain": "normalized",
                "pft_save_default": False,
            },
            "blending": "additive",
        }
        if req.active_scale is not None and len(req.active_scale) == 3:
            deconv_kwargs["scale"] = req.active_scale
        deconv_kwargs["contrast_limits"] = _contrast_limits(deconv_raw)
        yield StepLayerResult(deconv_raw.astype(np.float32), deconv_kwargs["name"], "image", deconv_kwargs)

        if req.add_refined_layer:
            refined_norm = deconv_norm.astype(np.float32) * mask.astype(np.float32)
            refined_raw = _denormalize(refined_norm, md_active["pft_norm_lo"], md_active["pft_norm_hi"])
            refined_kwargs = {
                "name": f"PFT | denoise | deconv + U-Net | {active_channel_name}",
                "metadata": {
                    **md_active,
                    "pft_step": "deconvolution_masked_by_unet",
                    "pft_psf_path": str(psf_path),
                    "pft_model_path": str(model_path),
                    "pft_variant": "raw",
                    "pft_is_normalized": False,
                    "pft_processing_domain": "normalized",
                    "pft_save_default": False,
                },
                "blending": "additive",
            }
            if req.active_scale is not None and len(req.active_scale) == 3:
                refined_kwargs["scale"] = req.active_scale
            refined_kwargs["contrast_limits"] = _contrast_limits(refined_raw)
            yield StepLayerResult(refined_raw.astype(np.float32), refined_kwargs["name"], "image", refined_kwargs)
        return

    raise ValueError(f"Unsupported denoising mode: {req.mode}")


class PFTDenoiseWidget(QWidget):
    """Represent pftdenoise widget and its associated operations."""
    def __init__(self, napari_viewer):
        """Initialize a ``PFTDenoiseWidget`` instance.

        Args:
            napari_viewer (Any): Value specifying napari viewer for the operation.

        Example:
            >>> instance = PFTDenoiseWidget(napari_viewer=...)
        """
        super().__init__()
        self.viewer = napari_viewer
        self.project_root = find_project_root(Path(__file__).resolve())
        self._worker = None

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("PFT denoising / restoration"))

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Auto", "2D: filter + U-Net", "3D: U-Net + deconvolution"])
        row_mode.addWidget(self.mode_combo)
        self.layout().addLayout(row_mode)

        row_dataset = QHBoxLayout()
        row_dataset.addWidget(QLabel("2D preset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Auto", "2d_time", "2d_wga_dapi"])
        row_dataset.addWidget(self.dataset_combo)
        self.layout().addLayout(row_dataset)

        row_filter = QHBoxLayout()
        row_filter.addWidget(QLabel("Filter percentile:"))
        self.filter_percentile = QDoubleSpinBox()
        self.filter_percentile.setRange(80.0, 100.0)
        self.filter_percentile.setSingleStep(0.1)
        self.filter_percentile.setValue(99.2)
        row_filter.addWidget(self.filter_percentile)
        self.layout().addLayout(row_filter)

        row_unet = QHBoxLayout()
        row_unet.addWidget(QLabel("U-Net threshold:"))
        self.unet_threshold = QDoubleSpinBox()
        self.unet_threshold.setRange(0.01, 0.99)
        self.unet_threshold.setSingleStep(0.05)
        self.unet_threshold.setValue(0.5)
        row_unet.addWidget(self.unet_threshold)
        row_unet.addWidget(QLabel("Patch:"))
        self.unet_patch = QSpinBox()
        self.unet_patch.setRange(64, 2048)
        self.unet_patch.setSingleStep(64)
        self.unet_patch.setValue(256)
        row_unet.addWidget(self.unet_patch)
        self.layout().addLayout(row_unet)

        row_model = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Optional U-Net model path; empty = automatic")
        row_model.addWidget(self.model_path)
        self.browse_model_btn = QPushButton("Model…")
        self.browse_model_btn.clicked.connect(self._browse_model)
        row_model.addWidget(self.browse_model_btn)
        self.layout().addLayout(row_model)

        row_deconv = QHBoxLayout()
        row_deconv.addWidget(QLabel("PSF model:"))
        self.psf_model = QComboBox()
        self.psf_model.addItems(["BW", "GL", "RW"])
        row_deconv.addWidget(self.psf_model)
        row_deconv.addWidget(QLabel("RL iter:"))
        self.rl_iterations = QSpinBox()
        self.rl_iterations.setRange(1, 100)
        self.rl_iterations.setValue(15)
        row_deconv.addWidget(self.rl_iterations)
        row_deconv.addWidget(QLabel("BG:"))
        self.rl_background = QDoubleSpinBox()
        self.rl_background.setRange(0.0, 1_000_000.0)
        self.rl_background.setDecimals(4)
        self.rl_background.setValue(0.0)
        row_deconv.addWidget(self.rl_background)
        self.layout().addLayout(row_deconv)

        row_psf = QHBoxLayout()
        self.psf_path = QLineEdit()
        self.psf_path.setPlaceholderText("Optional PSF .tif path; empty = automatic")
        row_psf.addWidget(self.psf_path)
        self.browse_psf_btn = QPushButton("PSF…")
        self.browse_psf_btn.clicked.connect(self._browse_psf)
        row_psf.addWidget(self.browse_psf_btn)
        self.layout().addLayout(row_psf)

        row_options = QHBoxLayout()
        self.add_mask_layer = QCheckBox("Add U-Net mask")
        self.add_mask_layer.setChecked(True)
        row_options.addWidget(self.add_mask_layer)
        self.add_refined_layer = QCheckBox("Add final refined layer")
        self.add_refined_layer.setChecked(True)
        row_options.addWidget(self.add_refined_layer)
        self.layout().addLayout(row_options)

        row_actions = QHBoxLayout()
        self.run_btn = QPushButton("Denoise active image")
        self.run_btn.clicked.connect(self.run_denoising)
        row_actions.addWidget(self.run_btn)

        self.norm_btn = QPushButton("Show normalized variants")
        self.norm_btn.clicked.connect(self.show_normalized_variants)
        row_actions.addWidget(self.norm_btn)
        self.layout().addLayout(row_actions)

        row_save = QHBoxLayout()
        self.save_btn = QPushButton("Save active layer as OME-Zarr…")
        self.save_btn.clicked.connect(self.save_active_layer_as_omezarr)
        row_save.addWidget(self.save_btn)
        self.layout().addLayout(row_save)

        row_view = QHBoxLayout()
        self.slice_btn = QPushButton("3D slice view")
        self.slice_btn.clicked.connect(lambda: self._set_ndisplay(2))
        row_view.addWidget(self.slice_btn)
        self.volume_btn = QPushButton("3D volume view")
        self.volume_btn.clicked.connect(lambda: self._set_ndisplay(3))
        row_view.addWidget(self.volume_btn)
        self.layout().addLayout(row_view)

        self.status = QLabel(
            "Raw images are opened by default. For 3D data, the default inspection mode is slice-by-slice. "
            "Use 'Show normalized variants' only when you want the additional normalized display layers. "
            "Denoising always operates internally on normalized data, but the default output layers are shown back in raw intensity scale."
        )
        self.status.setWordWrap(True)
        self.layout().addWidget(self.status)

    def _browse_model(self) -> None:
        """Return browse model for the supplied inputs.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._browse_model()
        """
        path, _ = QFileDialog.getOpenFileName(self, "Select U-Net model", str(self.project_root / "models"), "Keras model (*.keras *.h5);;All files (*)")
        if path:
            self.model_path.setText(path)

    def _browse_psf(self) -> None:
        """Return browse point-spread function for the supplied inputs.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._browse_psf()
        """
        path, _ = QFileDialog.getOpenFileName(self, "Select PSF file", str(self.project_root / "results" / "psf" / "generated"), "TIFF files (*.tif *.tiff);;All files (*)")
        if path:
            self.psf_path.setText(path)

    def _set_ndisplay(self, ndisplay: int) -> None:
        """Set ndisplay on the current object or configuration.

        Args:
            ndisplay (int): Numerical value controlling ndisplay.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._set_ndisplay(ndisplay=1)
        """
        self.viewer.dims.ndisplay = int(ndisplay)
        if int(ndisplay) == 2:
            self.status.setText("3D data are now shown slice-by-slice. Move through Z with the slider or mouse wheel.")
        else:
            self.status.setText("3D data are now shown in volume rendering mode.")

    def _show_error(self, title: str, message: str) -> None:
        """Display error to the user.

        Args:
            title (str): Title displayed on the generated figure or report section.
            message (str): Text value specifying message.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._show_error(title="title", message="message")
        """
        QMessageBox.critical(self, title, message)
        self.status.setText(f"{title}: {message}")

    def _active_layer(self):
        """Return active layer for the supplied inputs.

        Returns:
            Any: Result produced by the operation.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> result = instance._active_layer()
        """
        if not self.viewer.layers:
            return None
        return self.viewer.layers.selection.active or self.viewer.layers[-1]

    def _processing_family_layers(self, active_layer: Any) -> list[Any]:
        """Return processing family layers for the supplied inputs.

        Args:
            active_layer (Any): Value specifying active layer for the operation.

        Returns:
            list[Any]: Collection containing the generated or selected values.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> result = instance._processing_family_layers(active_layer=...)
        """
        out = []
        for layer in self.viewer.layers:
            if not _is_image_layer(layer) or not _is_pft_layer(layer):
                continue
            md = getattr(layer, "metadata", {}) or {}
            if md.get("pft_is_normalized", False):
                continue
            if layer is active_layer or _same_source(active_layer, layer):
                if np.ndim(layer.data) == np.ndim(active_layer.data):
                    out.append(layer)
        out = sorted(out, key=_channel_index)
        seen = set()
        unique = []
        for layer in out:
            if id(layer) not in seen:
                unique.append(layer)
                seen.add(id(layer))
        return unique

    def _normalization_family_layers(self, active_layer: Any) -> list[Any]:
        """Return normalization family layers for the supplied inputs.

        Args:
            active_layer (Any): Value specifying active layer for the operation.

        Returns:
            list[Any]: Collection containing the generated or selected values.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> result = instance._normalization_family_layers(active_layer=...)
        """
        active_md = getattr(active_layer, "metadata", {}) or {}
        active_step = active_md.get("pft_step")
        out = []
        for layer in self.viewer.layers:
            if not _is_image_layer(layer) or not _is_pft_layer(layer):
                continue
            md = getattr(layer, "metadata", {}) or {}
            if md.get("pft_is_normalized", False):
                continue
            if not (layer is active_layer or _same_source(active_layer, layer)):
                continue
            if md.get("pft_step") != active_step:
                continue
            if np.ndim(layer.data) != np.ndim(active_layer.data):
                continue
            out.append(layer)
        out = sorted(out, key=_channel_index)
        seen = set()
        unique = []
        for layer in out:
            if id(layer) not in seen:
                unique.append(layer)
                seen.add(id(layer))
        return unique

    def _build_request(self) -> DenoiseRequest:
        """Build request from the supplied inputs.

        Returns:
            DenoiseRequest: Result produced by the operation.

        Raises:
            RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> result = instance._build_request()
        """
        active = self._active_layer()
        if active is None or not hasattr(active, "data"):
            raise RuntimeError("No active image layer is selected.")
        axes = _layer_axes(active)
        mode_text = self.mode_combo.currentText()
        sibling_layers = self._processing_family_layers(active)

        if mode_text.startswith("2D"):
            mode = "2d"
        elif mode_text.startswith("3D"):
            mode = "3d"
        else:
            mode = "3d" if "z" in axes else "2d"

        dataset_text = self.dataset_combo.currentText()
        dataset = _infer_dataset_from_layers(active, sibling_layers) if dataset_text == "Auto" else dataset_text
        if mode == "3d":
            dataset = "3d_25d"

        selected_layers = sibling_layers if mode == "3d" or len(sibling_layers) > 1 else [active]
        packed_layers = []
        for layer in selected_layers:
            md = dict(getattr(layer, "metadata", {}) or {})
            md["axes"] = _layer_axes(layer)
            md["channel_name"] = _channel_name(layer)
            md["channel_index"] = _channel_index(layer)
            packed_layers.append((_channel_name(layer), _as_numpy(layer.data), md))

        model_text = self.model_path.text().strip().strip('"')
        psf_text = self.psf_path.text().strip().strip('"')
        active_md = getattr(active, "metadata", {}) or {}
        return DenoiseRequest(
            mode=mode,
            dataset=dataset,
            project_root=self.project_root,
            active_layer_name=_channel_name(active),
            active_channel_index=_channel_index(active),
            active_channel_name=_channel_name(active),
            active_axes=axes,
            active_scale=_layer_scale(active),
            active_colormap=getattr(getattr(active, "colormap", None), "name", None),
            source_ome_zarr=active_md.get("pft_ome_zarr_path"),
            layers=packed_layers,
            filter_high_percentile=float(self.filter_percentile.value()),
            unet_threshold=float(self.unet_threshold.value()),
            unet_patch=int(self.unet_patch.value()),
            model_path=Path(model_text) if model_text else None,
            psf_model=self.psf_model.currentText(),
            rl_iterations=int(self.rl_iterations.value()),
            rl_background=float(self.rl_background.value()),
            psf_path=Path(psf_text) if psf_text else None,
            add_mask_layer=bool(self.add_mask_layer.isChecked()),
            add_refined_layer=bool(self.add_refined_layer.isChecked()),
        )

    def run_denoising(self) -> None:
        """Run denoising using the supplied configuration.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance.run_denoising()
        """
        try:
            req = self._build_request()
        except Exception as exc:
            self._show_error("Cannot start denoising", f"{type(exc).__name__}: {exc}")
            return

        self.viewer.dims.ndisplay = 2
        self.run_btn.setEnabled(False)
        self.status.setText(
            f"Running {req.mode.upper()} denoising for dataset preset '{req.dataset}'. "
            "Processing is performed on normalized data. New raw-scale result layers will appear after each completed step."
        )

        if thread_worker is None:
            try:
                for result in _denoise_pipeline_worker(req):
                    self._add_result_layer(result)
                self._worker_done(None)
            except Exception as exc:
                self._worker_error(exc)
            return

        worker = thread_worker(_denoise_pipeline_worker)(req)
        worker.yielded.connect(self._add_result_layer)
        worker.returned.connect(self._worker_done)
        worker.errored.connect(self._worker_error)
        self._worker = worker
        worker.start()

    def _add_result_layer(self, result: StepLayerResult) -> None:
        """Add result layer to the current data structure.

        Args:
            result (StepLayerResult): Value specifying result for the operation.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._add_result_layer(result=...)
        """
        kwargs = dict(result.kwargs or {})
        name = kwargs.pop("name", result.name)
        if result.layer_type == "labels":
            self.viewer.add_labels(result.data, name=name, **kwargs)
        else:
            self.viewer.add_image(result.data, name=name, **kwargs)
        self.status.setText(f"Added layer: {name}")

    def _worker_done(self, _value: Any) -> None:
        """Return worker done for the supplied inputs.

        Args:
            _value (Any): Value to validate, transform, store, or forward.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._worker_done(_value=...)
        """
        self.run_btn.setEnabled(True)
        self.status.setText(
            "Denoising completed. Results are shown in raw intensity scale. "
            "If you also want the normalized display versions, select one result layer and click 'Show normalized variants'."
        )

    def _worker_error(self, exc: Exception) -> None:
        """Return worker error for the supplied inputs.

        Args:
            exc (Exception): Value specifying exc for the operation.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance._worker_error(exc=...)
        """
        self.run_btn.setEnabled(True)
        if isinstance(exc, tuple) and len(exc) >= 2:
            exc = exc[1]
        self._show_error("Denoising failed", f"{type(exc).__name__}: {exc}")

    def show_normalized_variants(self) -> None:
        """Display normalized variants to the user.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance.show_normalized_variants()
        """
        active = self._active_layer()
        if active is None or not _is_image_layer(active):
            self._show_error("Normalization failed", "Select an image layer first.")
            return
        targets = self._normalization_family_layers(active)
        if not targets:
            targets = [active]

        created = 0
        for layer in targets:
            arr = _as_numpy(layer.data)
            norm, lo, hi = _robust_normalize(arr)
            md = dict(getattr(layer, "metadata", {}) or {})
            new_name = f"{layer.name} | normalized"
            if new_name in self.viewer.layers:
                continue
            kwargs: dict[str, Any] = {
                "name": new_name,
                "metadata": {
                    **md,
                    "pft_variant": "normalized",
                    "pft_is_normalized": True,
                    "pft_normalized_from": str(layer.name),
                    "pft_normalized_lo": float(lo),
                    "pft_normalized_hi": float(hi),
                    "pft_save_default": False,
                },
                "blending": getattr(layer, "blending", "additive"),
                "contrast_limits": (0.0, 1.0),
            }
            scale = _layer_scale(layer)
            if scale is not None and len(scale) == norm.ndim:
                kwargs["scale"] = scale
            try:
                kwargs["colormap"] = layer.colormap.name
            except Exception:
                pass
            self.viewer.add_image(norm.astype(np.float32), **kwargs)
            created += 1
        self.status.setText(f"Added {created} normalized layer(s). Raw layers remain unchanged.")

    def save_active_layer_as_omezarr(self) -> None:
        """Save active layer as OME-Zarr to persistent storage.

        Example:
            >>> instance = PFTDenoiseWidget(...)
            >>> instance.save_active_layer_as_omezarr()
        """
        layer = self._active_layer()
        if layer is None or not hasattr(layer, "data"):
            self._show_error("Save failed", "No active layer is selected.")
            return
        parent = QFileDialog.getExistingDirectory(self, "Select output parent folder", str(self.project_root / "results" / "Napari_zarr"))
        if not parent:
            return
        data = _as_numpy(layer.data)
        axes = _layer_axes(layer)
        if len(axes) != data.ndim:
            axes = "yx" if data.ndim == 2 else "zyx" if data.ndim == 3 else "czyx" if data.ndim == 4 else "".join(f"d{i}" for i in range(data.ndim))
        out_zarr = Path(parent) / f"{_safe_name(layer.name)}.ome.zarr"
        try:
            save_ome_zarr(out_zarr, data, axes, meta=None, overwrite=True, pyramid_3d=("z" in axes), pyramid_max_layer=2 if "z" in axes else 0)
        except Exception as exc:
            self._show_error("Save failed", f"{type(exc).__name__}: {exc}")
            return
        self.status.setText(f"Saved active layer as OME-Zarr: {out_zarr}")
