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
    data: np.ndarray
    name: str
    layer_type: str = "image"
    kwargs: dict[str, Any] | None = None


@dataclass
class SegmentationRequest:
    mode: str
    model_family: str
    dataset: str
    project_root: Path
    model_type: str
    custom_model_path: Path | None
    gpu: bool
    diameter: float | None
    flow_threshold: float
    cellprob_threshold: float
    mask_threshold: float
    min_size: int
    stitch_threshold: float
    active_layer_name: str
    active_scale: tuple[float, ...] | None
    source_ome_zarr: str | None
    layers: list[tuple[str, np.ndarray, dict[str, Any]]]
    show_merged_input: bool


def make_segmentation_widget(napari_viewer):
    return PFTSegmentationWidget(napari_viewer)


def _safe_name(text: str) -> str:
    bad = r'<>:"/\\|?*• '
    cleaned = "".join("_" if ch in bad else ch for ch in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "layer"


def _as_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def _layer_axes(layer: Any) -> str:
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
    scale = getattr(layer, "scale", None)
    if scale is None:
        return None
    try:
        return tuple(float(v) for v in scale)
    except Exception:
        return None


def _contrast_limits(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> tuple[float, float]:
    x = np.asarray(data)
    if x.size == 0:
        return 0.0, 1.0
    flat = x.reshape(-1)
    if flat.size > 2_000_000:
        flat = flat[:: max(1, flat.size // 2_000_000)]
    lo, hi = np.percentile(flat, [p_low, p_high])
    lo = float(lo)
    hi = float(hi if hi > lo else lo + 1.0)
    return lo, hi


def _normalize_channel(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = _contrast_limits(x, p_low, p_high)
    y = (x - np.float32(lo)) / np.float32(hi - lo + 1e-8)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _normalize_merged(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2:
        return _normalize_channel(x)
    if x.ndim == 3:
        # 2D multi-channel HWC or 3D single-channel ZYX
        if x.shape[-1] <= 6 and x.shape[0] > 6 and x.shape[1] > 6:
            out = np.empty_like(x, dtype=np.float32)
            for c in range(x.shape[-1]):
                out[..., c] = _normalize_channel(x[..., c])
            return out
        return _normalize_channel(x)
    if x.ndim == 4:
        out = np.empty_like(x, dtype=np.float32)
        for c in range(x.shape[-1]):
            out[..., c] = _normalize_channel(x[..., c])
        return out
    raise ValueError(f"Unsupported merged array shape: {x.shape}")


def _is_image_layer(layer: Any) -> bool:
    return hasattr(layer, "data") and hasattr(layer, "metadata") and layer.__class__.__name__.lower() != "labels"


def _is_pft_layer(layer: Any) -> bool:
    name = str(getattr(layer, "name", ""))
    md = getattr(layer, "metadata", {}) or {}
    return name.startswith("PFT |") or any(str(k).startswith("pft_") for k in md)


def _same_source(a: Any, b: Any) -> bool:
    ma = getattr(a, "metadata", {}) or {}
    mb = getattr(b, "metadata", {}) or {}
    za = ma.get("pft_ome_zarr_path") or ma.get("pft_source_path")
    zb = mb.get("pft_ome_zarr_path") or mb.get("pft_source_path")
    return bool(za and zb and str(za) == str(zb))


def _same_variant_and_step(a: Any, b: Any) -> bool:
    ma = getattr(a, "metadata", {}) or {}
    mb = getattr(b, "metadata", {}) or {}
    return (
        ma.get("pft_variant") == mb.get("pft_variant")
        and ma.get("pft_step") == mb.get("pft_step")
        and ma.get("pft_is_normalized", False) == mb.get("pft_is_normalized", False)
    )


def _channel_index(layer: Any) -> int:
    md = getattr(layer, "metadata", {}) or {}
    try:
        return int(md.get("pft_channel_index", 0))
    except Exception:
        return 0


def _channel_name(layer: Any) -> str:
    md = getattr(layer, "metadata", {}) or {}
    return str(md.get("pft_channel_name") or getattr(layer, "name", "channel"))


def _infer_dataset(active_layer: Any, family_layers: list[Any]) -> str:
    axes = _layer_axes(active_layer)
    if "z" in axes:
        return "3d"
    if len(family_layers) >= 2:
        return "2d_wga_dapi"
    return "2d_time"


def _merge_for_segmentation(layers: list[tuple[str, np.ndarray, dict[str, Any]]], mode: str) -> np.ndarray:
    arrays = [np.asarray(arr, dtype=np.float32) for _, arr, _ in layers]
    if not arrays:
        raise ValueError("No layers available for segmentation.")

    first = arrays[0]
    if mode == "2d":
        if first.ndim != 2:
            raise ValueError(f"2D segmentation expects YX layers. Got shape={first.shape}")
        if len(arrays) == 1:
            return _normalize_merged(first)
        merged = np.stack(arrays, axis=-1)  # HWC
        return _normalize_merged(merged)

    if mode == "3d":
        if first.ndim != 3:
            raise ValueError(f"3D segmentation expects ZYX layers. Got shape={first.shape}")
        if len(arrays) == 1:
            return _normalize_merged(first)  # ZYX
        merged = np.stack(arrays, axis=-1)  # ZYXC
        return _normalize_merged(merged)

    raise ValueError(f"Unsupported mode={mode}")


def _build_cellpose_model(req: SegmentationRequest):
    from cellpose import models

    if req.custom_model_path is not None:
        return models.CellposeModel(gpu=req.gpu, pretrained_model=str(req.custom_model_path))
    return models.Cellpose(gpu=req.gpu, model_type=req.model_type)


def _run_cellpose(req: SegmentationRequest, merged: np.ndarray) -> np.ndarray:
    model = _build_cellpose_model(req)
    diameter = None if req.diameter is None or req.diameter <= 0 else float(req.diameter)
    kwargs = dict(
        diameter=diameter,
        flow_threshold=float(req.flow_threshold),
        cellprob_threshold=float(req.cellprob_threshold),
        min_size=int(req.min_size),
        normalize=True,
    )

    if req.mode == "2d":
        if merged.ndim == 3 and merged.shape[-1] <= 6:
            kwargs["channel_axis"] = -1
        masks, *_ = model.eval(merged, **kwargs)
        return np.asarray(masks, dtype=np.uint16)

    # 3D
    if merged.ndim == 4 and merged.shape[-1] <= 6:
        kwargs.update(dict(do_3D=True, z_axis=0, channel_axis=-1, stitch_threshold=float(req.stitch_threshold)))
    else:
        kwargs.update(dict(do_3D=True, z_axis=0, channel_axis=None, stitch_threshold=float(req.stitch_threshold)))
    masks, *_ = model.eval(merged, **kwargs)
    return np.asarray(masks, dtype=np.uint16)


def _build_omnipose_model(req: SegmentationRequest):
    try:
        from cellpose_omni import models
    except Exception:
        from cellpose import models

    if req.custom_model_path is not None:
        try:
            return models.CellposeModel(gpu=req.gpu, pretrained_model=str(req.custom_model_path))
        except TypeError:
            return models.CellposeModel(gpu=req.gpu, pretrained_model=str(req.custom_model_path), nchan=1)
    try:
        return models.Cellpose(gpu=req.gpu, model_type=req.model_type)
    except TypeError:
        return models.CellposeModel(gpu=req.gpu, model_type=req.model_type)


def _run_omnipose(req: SegmentationRequest, merged: np.ndarray) -> np.ndarray:
    model = _build_omnipose_model(req)
    diameter = None if req.diameter is None or req.diameter <= 0 else float(req.diameter)

    if req.mode == "2d":
        res = model.eval(
            merged,
            channels=[0, 0],
            diameter=diameter,
            mask_threshold=float(req.mask_threshold),
            flow_threshold=float(req.flow_threshold),
            min_size=int(req.min_size),
        )
        masks = res[0] if isinstance(res, tuple) else res
        return np.asarray(masks, dtype=np.uint16)

    # 3D fallback: slice-wise Omnipose, consistent with current core approach.
    if merged.ndim == 4:
        z_total = merged.shape[0]
        labels = np.zeros(merged.shape[:3], dtype=np.uint16)
        for z in range(z_total):
            res = model.eval(
                merged[z],
                channels=[0, 0],
                diameter=diameter,
                mask_threshold=float(req.mask_threshold),
                flow_threshold=float(req.flow_threshold),
                min_size=int(req.min_size),
            )
            masks = res[0] if isinstance(res, tuple) else res
            labels[z] = np.asarray(masks, dtype=np.uint16)
        return labels

    z_total = merged.shape[0]
    labels = np.zeros_like(merged, dtype=np.uint16)
    for z in range(z_total):
        res = model.eval(
            merged[z],
            channels=[0, 0],
            diameter=diameter,
            mask_threshold=float(req.mask_threshold),
            flow_threshold=float(req.flow_threshold),
            min_size=int(req.min_size),
        )
        masks = res[0] if isinstance(res, tuple) else res
        labels[z] = np.asarray(masks, dtype=np.uint16)
    return labels


def _segmentation_worker(req: SegmentationRequest) -> Iterable[StepLayerResult]:
    merged = _merge_for_segmentation(req.layers, req.mode)
    axes = "yx" if req.mode == "2d" else "zyx"

    if req.show_merged_input:
        merged_kwargs = {
            "name": "PFT | segmentation | merged input",
            "metadata": {
                "pft_step": "segmentation_input_merged",
                "pft_dataset": req.dataset,
                "pft_model_family": req.model_family,
                "pft_variant": "normalized",
                "pft_is_normalized": True,
                "pft_axes": axes if merged.ndim < 4 else ("yxc" if req.mode == "2d" else "zyxc"),
                "pft_ome_zarr_path": req.source_ome_zarr,
                "pft_save_default": False,
            },
            "blending": "additive",
        }
        if req.active_scale is not None:
            if req.mode == "2d" and merged.ndim == 2 and len(req.active_scale) == 2:
                merged_kwargs["scale"] = req.active_scale
            elif req.mode == "2d" and merged.ndim == 3 and len(req.active_scale) == 2:
                pass
            elif req.mode == "3d" and merged.ndim == 3 and len(req.active_scale) == 3:
                merged_kwargs["scale"] = req.active_scale
            elif req.mode == "3d" and merged.ndim == 4 and len(req.active_scale) == 3:
                pass
        if merged.dtype != np.uint16:
            merged_kwargs["contrast_limits"] = (0.0, 1.0)
        yield StepLayerResult(merged.astype(np.float32), merged_kwargs["name"], "image", merged_kwargs)

    if req.model_family == "cellpose":
        labels = _run_cellpose(req, merged)
    else:
        labels = _run_omnipose(req, merged)

    label_kwargs = {
        "name": f"PFT | segmentation | {req.model_family} mask",
        "metadata": {
            "pft_step": f"segmentation_{req.model_family}",
            "pft_dataset": req.dataset,
            "pft_model_family": req.model_family,
            "pft_model_type": req.model_type,
            "pft_model_path": str(req.custom_model_path) if req.custom_model_path else None,
            "pft_source_layer": req.active_layer_name,
            "pft_axes": axes,
            "pft_ome_zarr_path": req.source_ome_zarr,
            "pft_save_default": False,
        },
    }
    if req.active_scale is not None and len(req.active_scale) == labels.ndim:
        label_kwargs["scale"] = req.active_scale
    yield StepLayerResult(labels.astype(np.uint16), label_kwargs["name"], "labels", label_kwargs)


class PFTSegmentationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.project_root = find_project_root(Path(__file__).resolve())
        self._worker = None

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("PFT segmentation"))

        row_family = QHBoxLayout()
        row_family.addWidget(QLabel("Model family:"))
        self.family_combo = QComboBox()
        self.family_combo.addItems(["Cellpose", "Omnipose"])
        self.family_combo.currentTextChanged.connect(self._update_family_defaults)
        row_family.addWidget(self.family_combo)
        self.layout().addLayout(row_family)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Auto", "2D", "3D"])
        row_mode.addWidget(self.mode_combo)
        self.layout().addLayout(row_mode)

        row_dataset = QHBoxLayout()
        row_dataset.addWidget(QLabel("Dataset preset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Auto", "2d_time", "2d_wga_dapi", "3d"])
        row_dataset.addWidget(self.dataset_combo)
        self.layout().addLayout(row_dataset)

        row_model_type = QHBoxLayout()
        row_model_type.addWidget(QLabel("Built-in model:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["cyto", "cyto2", "cpsam"])
        row_model_type.addWidget(self.model_type_combo)
        self.layout().addLayout(row_model_type)

        row_custom = QHBoxLayout()
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setPlaceholderText("Optional custom Cellpose / Omnipose model path")
        row_custom.addWidget(self.custom_model_path)
        self.browse_model_btn = QPushButton("Model…")
        self.browse_model_btn.clicked.connect(self._browse_model)
        row_custom.addWidget(self.browse_model_btn)
        self.layout().addLayout(row_custom)

        row_num = QHBoxLayout()
        row_num.addWidget(QLabel("Diameter:"))
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0.0, 500.0)
        self.diameter.setSingleStep(1.0)
        self.diameter.setValue(0.0)
        self.diameter.setSpecialValueText("Auto")
        row_num.addWidget(self.diameter)
        row_num.addWidget(QLabel("Min size:"))
        self.min_size = QSpinBox()
        self.min_size.setRange(0, 10000)
        self.min_size.setValue(15)
        row_num.addWidget(self.min_size)
        self.layout().addLayout(row_num)

        row_thr1 = QHBoxLayout()
        row_thr1.addWidget(QLabel("Flow threshold:"))
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(-10.0, 10.0)
        self.flow_threshold.setDecimals(3)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setValue(0.4)
        row_thr1.addWidget(self.flow_threshold)
        row_thr1.addWidget(QLabel("Cellprob (Cellpose):"))
        self.cellprob_threshold = QDoubleSpinBox()
        self.cellprob_threshold.setRange(-10.0, 10.0)
        self.cellprob_threshold.setDecimals(3)
        self.cellprob_threshold.setSingleStep(0.1)
        self.cellprob_threshold.setValue(0.0)
        row_thr1.addWidget(self.cellprob_threshold)
        self.layout().addLayout(row_thr1)

        row_thr2 = QHBoxLayout()
        row_thr2.addWidget(QLabel("Mask threshold (Omnipose):"))
        self.mask_threshold = QDoubleSpinBox()
        self.mask_threshold.setRange(-10.0, 10.0)
        self.mask_threshold.setDecimals(3)
        self.mask_threshold.setSingleStep(0.1)
        self.mask_threshold.setValue(0.0)
        row_thr2.addWidget(self.mask_threshold)
        row_thr2.addWidget(QLabel("Stitch threshold (3D Cellpose):"))
        self.stitch_threshold = QDoubleSpinBox()
        self.stitch_threshold.setRange(0.0, 1.0)
        self.stitch_threshold.setDecimals(3)
        self.stitch_threshold.setSingleStep(0.05)
        self.stitch_threshold.setValue(0.0)
        row_thr2.addWidget(self.stitch_threshold)
        self.layout().addLayout(row_thr2)

        row_opts = QHBoxLayout()
        self.gpu_box = QCheckBox("Use GPU")
        self.gpu_box.setChecked(True)
        row_opts.addWidget(self.gpu_box)
        self.show_merged_box = QCheckBox("Show merged segmentation input")
        self.show_merged_box.setChecked(True)
        row_opts.addWidget(self.show_merged_box)
        self.layout().addLayout(row_opts)

        row_actions = QHBoxLayout()
        self.run_btn = QPushButton("Run segmentation on active denoised image")
        self.run_btn.clicked.connect(self.run_segmentation)
        row_actions.addWidget(self.run_btn)
        self.save_btn = QPushButton("Save active layer as OME-Zarr…")
        self.save_btn.clicked.connect(self.save_active_layer_as_omezarr)
        row_actions.addWidget(self.save_btn)
        self.layout().addLayout(row_actions)

        self.status = QLabel(
            "Select a denoised PFT image layer. The widget will collect matching same-source channel layers, "
            "merge them into one segmentation input image, run Cellpose or Omnipose, and add the label mask as a new napari layer."
        )
        self.status.setWordWrap(True)
        self.layout().addWidget(self.status)

        self._update_family_defaults(self.family_combo.currentText())

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select custom segmentation model",
            str(self.project_root / "models"),
            "Model files (*.pth *.pt *.npy *.h5 *.keras);;All files (*)",
        )
        if path:
            self.custom_model_path.setText(path)

    def _update_family_defaults(self, family: str) -> None:
        self.model_type_combo.clear()
        if family.lower().startswith("omni"):
            self.model_type_combo.addItems(["cyto2_omni", "bact_phase_omni", "cyto"])
        else:
            self.model_type_combo.addItems(["cpsam", "cyto2", "cyto"])

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.status.setText(f"{title}: {message}")

    def _active_layer(self):
        if not self.viewer.layers:
            return None
        return self.viewer.layers.selection.active or self.viewer.layers[-1]

    def _collect_family_layers(self, active_layer: Any) -> list[Any]:
        if active_layer is None or not _is_image_layer(active_layer):
            return []
        md_active = getattr(active_layer, "metadata", {}) or {}
        candidates = []
        for layer in self.viewer.layers:
            if not _is_image_layer(layer) or not _is_pft_layer(layer):
                continue
            md = getattr(layer, "metadata", {}) or {}
            if md.get("pft_is_normalized", False):
                continue
            if not (layer is active_layer or _same_source(active_layer, layer)):
                continue
            if np.ndim(layer.data) != np.ndim(active_layer.data):
                continue
            if not _same_variant_and_step(active_layer, layer):
                continue
            candidates.append(layer)

        candidates = sorted(candidates, key=_channel_index)
        seen = set()
        unique = []
        for layer in candidates:
            if id(layer) not in seen:
                unique.append(layer)
                seen.add(id(layer))
        return unique

    def _build_request(self) -> SegmentationRequest:
        active = self._active_layer()
        if active is None or not _is_image_layer(active):
            raise RuntimeError("No active image layer is selected.")

        family_layers = self._collect_family_layers(active)
        if not family_layers:
            family_layers = [active]

        active_axes = _layer_axes(active)
        mode_text = self.mode_combo.currentText()
        if mode_text == "2D":
            mode = "2d"
        elif mode_text == "3D":
            mode = "3d"
        else:
            mode = "3d" if "z" in active_axes else "2d"

        dataset_text = self.dataset_combo.currentText()
        dataset = _infer_dataset(active, family_layers) if dataset_text == "Auto" else dataset_text
        family = self.family_combo.currentText().strip().lower()
        family = "omnipose" if family.startswith("omni") else "cellpose"

        packed_layers = []
        for layer in family_layers:
            md = dict(getattr(layer, "metadata", {}) or {})
            md["axes"] = _layer_axes(layer)
            md["channel_name"] = _channel_name(layer)
            md["channel_index"] = _channel_index(layer)
            packed_layers.append((_channel_name(layer), _as_numpy(layer.data), md))

        model_path_text = self.custom_model_path.text().strip().strip('"')
        active_md = getattr(active, "metadata", {}) or {}

        return SegmentationRequest(
            mode=mode,
            model_family=family,
            dataset=dataset,
            project_root=self.project_root,
            model_type=self.model_type_combo.currentText(),
            custom_model_path=Path(model_path_text) if model_path_text else None,
            gpu=bool(self.gpu_box.isChecked()),
            diameter=None if float(self.diameter.value()) <= 0 else float(self.diameter.value()),
            flow_threshold=float(self.flow_threshold.value()),
            cellprob_threshold=float(self.cellprob_threshold.value()),
            mask_threshold=float(self.mask_threshold.value()),
            min_size=int(self.min_size.value()),
            stitch_threshold=float(self.stitch_threshold.value()),
            active_layer_name=str(active.name),
            active_scale=_layer_scale(active),
            source_ome_zarr=active_md.get("pft_ome_zarr_path"),
            layers=packed_layers,
            show_merged_input=bool(self.show_merged_box.isChecked()),
        )

    def run_segmentation(self) -> None:
        try:
            req = self._build_request()
        except Exception as exc:
            self._show_error("Cannot start segmentation", f"{type(exc).__name__}: {exc}")
            return

        self.viewer.dims.ndisplay = 2
        self.run_btn.setEnabled(False)
        self.status.setText(
            f"Running {req.model_family} segmentation in {req.mode.upper()} mode on merged denoised input. "
            "A merged input layer and the label mask layer will be added to napari."
        )

        if thread_worker is None:
            try:
                for result in _segmentation_worker(req):
                    self._add_result_layer(result)
                self._worker_done(None)
            except Exception as exc:
                self._worker_error(exc)
            return

        worker = thread_worker(_segmentation_worker)(req)
        worker.yielded.connect(self._add_result_layer)
        worker.returned.connect(self._worker_done)
        worker.errored.connect(self._worker_error)
        self._worker = worker
        worker.start()

    def _add_result_layer(self, result: StepLayerResult) -> None:
        kwargs = dict(result.kwargs or {})
        name = kwargs.pop("name", result.name)
        if result.layer_type == "labels":
            self.viewer.add_labels(result.data, name=name, **kwargs)
        else:
            self.viewer.add_image(result.data, name=name, **kwargs)
        self.status.setText(f"Added layer: {name}")

    def _worker_done(self, _value: Any) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(
            "Segmentation completed. The merged input and label mask are now visible as napari layers. "
            "You can select the label layer and save it manually if needed."
        )

    def _worker_error(self, exc: Exception) -> None:
        self.run_btn.setEnabled(True)
        if isinstance(exc, tuple) and len(exc) >= 2:
            exc = exc[1]
        self._show_error("Segmentation failed", f"{type(exc).__name__}: {exc}")

    def save_active_layer_as_omezarr(self) -> None:
        layer = self._active_layer()
        if layer is None or not hasattr(layer, "data"):
            self._show_error("Save failed", "No active layer is selected.")
            return
        parent = QFileDialog.getExistingDirectory(
            self,
            "Select output parent folder",
            str(self.project_root / "results" / "Napari_zarr"),
        )
        if not parent:
            return
        data = _as_numpy(layer.data)
        axes = _layer_axes(layer)
        if len(axes) != data.ndim:
            axes = "yx" if data.ndim == 2 else "zyx" if data.ndim == 3 else "".join(f"d{i}" for i in range(data.ndim))
        out_zarr = Path(parent) / f"{_safe_name(layer.name)}.ome.zarr"
        try:
            save_ome_zarr(
                out_zarr,
                data,
                axes,
                meta=None,
                overwrite=True,
                pyramid_3d=("z" in axes),
                pyramid_max_layer=2 if "z" in axes else 0,
            )
        except Exception as exc:
            self._show_error("Save failed", f"{type(exc).__name__}: {exc}")
            return
        self.status.setText(f"Saved active layer as OME-Zarr: {out_zarr}")
