"""
Run, train, and tune 2D instance-segmentation models on prepared PFT inputs.
 results/training_files/segmentation/<dataset>/<source_mode>/<sample>/mask.tif

Models::

    models/segmentation/<family>/<dataset>/<source_mode>/<run_name>/

Predictions::

    results/segmentation_predictions/<family>/<dataset>/<source_mode>/
        <model_name>/<sample>/labels.ome.zarr
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
import csv
import json
import inspect
import random
import shutil
import sys

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.segmentation_input_core import (
    SOURCE_MODES_BY_DATASET,
    segmentation_input_root,
    segmentation_mask_root,
    segmentation_model_root,
    segmentation_prediction_root,
)

ModelFamily = Literal["cellpose", "omnipose", "stardist"]
MODEL_FAMILIES: tuple[str, ...] = ("cellpose", "omnipose", "stardist")


@dataclass(frozen=True)
class PreparedInputItem:
    """One prepared input and its canonical sample key."""

    sample_key: str
    input_zarr: Path
    training_mask: Path


@dataclass
class PredictionConfig:
    """Configuration shared by the three supported prediction families."""

    project_root: Path
    family: str
    dataset: str
    source_mode: str
    model: str | Path | None = None
    model_name: str | None = None
    gpu: bool = True
    selected_sample_keys: set[str] | None = None
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    mask_threshold: float = 0.0
    min_size: int = 15
    batch_size: int = 8
    prob_thresh: float | None = None
    nms_thresh: float | None = None


@dataclass
class TrainingConfig:
    """Configuration shared by initial model training workflows."""

    project_root: Path
    family: str
    dataset: str
    source_mode: str
    run_name: str
    pretrained_model: str | Path | None = None
    gpu: bool = True
    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    validation_fraction: float = 0.2
    seed: int = 1337
    save_every: int = 50
    min_train_masks: int = 1
    n_rays: int = 32
    grid: int = 2
    steps_per_epoch: int = 100
    patch_size: tuple[int, int] = (256, 256)
    omnipose_nclasses: int = 3
    extra_cli: list[str] = field(default_factory=list)


@dataclass
class TuneConfig:
    """Configuration for parameter tuning against manual instance masks."""

    project_root: Path
    family: str
    dataset: str
    source_mode: str
    model: str | Path | None = None
    model_name: str | None = None
    gpu: bool = True
    selected_sample_keys: set[str] | None = None
    instance_iou_threshold: float = 0.5
    diameter_values: tuple[float | None, ...] = (None,)
    flow_threshold_values: tuple[float, ...] = (0.2, 0.4, 0.6)
    cellprob_threshold_values: tuple[float, ...] = (-1.0, 0.0, 1.0)
    mask_threshold_values: tuple[float, ...] = (-1.0, 0.0, 1.0)
    prob_thresh_values: tuple[float, ...] = (0.3, 0.5, 0.7)
    nms_thresh_values: tuple[float, ...] = (0.3, 0.4, 0.5)
    min_size: int = 15
    batch_size: int = 8


def _validate_family_dataset(family: str, dataset: str, source_mode: str) -> tuple[str, str, str]:
    family = family.lower()
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {family!r}")
    if dataset not in SOURCE_MODES_BY_DATASET:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if source_mode not in SOURCE_MODES_BY_DATASET[dataset]:
        raise ValueError(
            f"Source mode {source_mode!r} is invalid for {dataset!r}: "
            f"{SOURCE_MODES_BY_DATASET[dataset]}"
        )
    return family, dataset, source_mode


def list_prepared_inputs(
    project_root: Path,
    dataset: str,
    source_mode: str,
) -> list[PreparedInputItem]:
    """List prepared inputs and corresponding manual mask paths."""
    _validate_family_dataset("cellpose", dataset, source_mode)
    input_root = segmentation_input_root(project_root, dataset, source_mode)
    mask_root = segmentation_mask_root(project_root, dataset, source_mode)
    paths = sorted(input_root.glob("**/segmentation_input.ome.zarr"))
    items = []
    for path in paths:
        relative = path.parent.relative_to(input_root)
        items.append(
            PreparedInputItem(
                sample_key=relative.as_posix(),
                input_zarr=path,
                training_mask=mask_root / relative / "mask.tif",
            )
        )
    return items


def _remove_singleton_nonspatial(
    array: np.ndarray,
    axes: str,
) -> tuple[np.ndarray, str]:
    for axis in tuple(axes):
        if axis not in {"c", "y", "x"}:
            index = axes.index(axis)
            if array.shape[index] != 1:
                raise ValueError(
                    f"Instance segmentation currently requires one 2D frame; "
                    f"received {axis}={array.shape[index]}, axes={axes}, shape={array.shape}"
                )
            array = np.take(array, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]
    return array, axes


def load_prepared_image(path: Path) -> np.ndarray:
    """Load one prepared OME-Zarr as YX or YXC without renormalizing it."""
    array, axes = load_ome_zarr(path, level=0, as_numpy=True)
    image = np.asarray(array)
    axes = str(axes).lower()
    image, axes = _remove_singleton_nonspatial(image, axes)
    if image.dtype != np.float32:
        raise TypeError(f"Prepared input must be float32, received {image.dtype}: {path}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Prepared input contains NaN or infinite values: {path}")
    if image.size == 0 or float(np.min(image)) < 0.0 or float(np.max(image)) > 1.0:
        raise ValueError(f"Prepared input must be finite and in [0,1]: {path}")
    if np.count_nonzero(image) == 0:
        raise ValueError(f"Prepared input is completely black: {path}")

    if axes == "yx":
        return image
    if set(axes) != {"c", "y", "x"} or len(axes) != 3:
        raise ValueError(f"Expected YX or CYX input, received axes={axes}, shape={image.shape}")
    return np.transpose(image, [axes.index("y"), axes.index("x"), axes.index("c")])


def load_instance_mask(path: Path, expected_yx: tuple[int, int]) -> np.ndarray:
    """Load and validate one non-negative integer instance mask."""
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.squeeze(np.asarray(tiff.imread(path)))
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(
            f"Mask shape {mask.shape} does not match expected YX {expected_yx}: {path}"
        )
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise TypeError(f"Mask must contain non-negative integer labels: {path}")
    if np.count_nonzero(mask) == 0:
        raise ValueError(f"Mask contains no cell instances: {path}")
    return mask.astype(np.int32, copy=False)


def collect_training_data(
    project_root: Path,
    dataset: str,
    source_mode: str,
    *,
    selected_sample_keys: set[str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load all prepared image/mask pairs with strict instance-label checks."""
    items = list_prepared_inputs(project_root, dataset, source_mode)
    if selected_sample_keys is not None:
        items = [item for item in items if item.sample_key in selected_sample_keys]
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    names: list[str] = []
    missing: list[str] = []
    for item in items:
        if not item.training_mask.is_file():
            missing.append(item.sample_key)
            continue
        image = load_prepared_image(item.input_zarr)
        mask = load_instance_mask(item.training_mask, image.shape[:2])
        images.append(image)
        masks.append(mask)
        names.append(item.sample_key)
    if not images:
        message = "No valid prepared image/mask training pairs were found."
        if missing:
            message += f" Missing masks: {missing}"
        raise RuntimeError(message)
    return images, masks, names


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _model_display_name(cfg: PredictionConfig) -> str:
    if cfg.model_name:
        return cfg.model_name
    if cfg.model is None:
        defaults = {
            "cellpose": "cellpose_sam_default",
            "omnipose": "bact_fluor_omni",
            "stardist": "2D_versatile_fluo",
        }
        return defaults[cfg.family]
    return Path(str(cfg.model)).stem


def _select_cellpose_builtin(models_module: Any) -> str:
    """Choose a Cellpose-SAM model name supported by the installed version."""
    available = set(getattr(models_module, "MODEL_NAMES", ()) or ())
    for candidate in ("cpsam_v2", "cpsam"):
        if not available or candidate in available:
            return candidate
    return "cpsam"


def _supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Filter keyword arguments for compatibility with installed library versions."""
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return values
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _load_prediction_model(cfg: PredictionConfig):
    family, _dataset, _mode = _validate_family_dataset(
        cfg.family, cfg.dataset, cfg.source_mode
    )
    if family == "cellpose":
        from cellpose import models

        pretrained = (
            str(cfg.model)
            if cfg.model is not None
            else _select_cellpose_builtin(models)
        )
        return models.CellposeModel(gpu=cfg.gpu, pretrained_model=pretrained)

    if family == "omnipose":
        from cellpose_omni import models

        if cfg.model is not None:
            try:
                return models.CellposeModel(
                    gpu=cfg.gpu,
                    pretrained_model=str(cfg.model),
                    nchan=2 if cfg.dataset == "2d_wga_dapi" else 1,
                    nclasses=3,
                    dim=2,
                )
            except TypeError:
                return models.CellposeModel(gpu=cfg.gpu, pretrained_model=str(cfg.model))
        model_type = cfg.model_name or "bact_fluor_omni"
        try:
            return models.Cellpose(
                gpu=cfg.gpu,
                model_type=model_type,
                nchan=2 if cfg.dataset == "2d_wga_dapi" else 1,
                nclasses=3,
                dim=2,
            )
        except TypeError:
            return models.Cellpose(gpu=cfg.gpu, model_type=model_type)

    model_value = str(cfg.model) if cfg.model is not None else None
    if model_value and Path(model_value).exists():
        model_dir = Path(model_value).resolve()
        if model_dir.is_file():
            model_dir = model_dir.parent
        from stardist.models import StarDist2D

        return StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained(model_value or "2D_versatile_fluo")


def _cellpose_predict(model, image: np.ndarray, cfg: PredictionConfig) -> np.ndarray:
    kwargs = {
        "diameter": cfg.diameter,
        "flow_threshold": cfg.flow_threshold,
        "cellprob_threshold": cfg.cellprob_threshold,
        "min_size": cfg.min_size,
        "batch_size": cfg.batch_size,
        "normalize": False,
        "channel_axis": -1 if image.ndim == 3 else None,
    }
    result = model.eval(image, **kwargs)
    masks = result[0] if isinstance(result, tuple) else result
    return np.asarray(masks, dtype=np.int32)


def _omnipose_predict(model, image: np.ndarray, cfg: PredictionConfig) -> np.ndarray:
    channels = [0, 0] if image.ndim == 2 or image.shape[-1] == 1 else [2, 1]
    kwargs = {
        "diameter": cfg.diameter,
        "channels": channels,
        "omni": True,
        "mask_threshold": cfg.mask_threshold,
        "flow_threshold": cfg.flow_threshold,
        "min_size": cfg.min_size,
        "normalize": False,
    }
    result = model.eval([image], **kwargs)
    masks = result[0]
    if isinstance(masks, (list, tuple)):
        masks = masks[0]
    return np.asarray(masks, dtype=np.int32)


def _stardist_predict(model, image: np.ndarray, cfg: PredictionConfig) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if cfg.prob_thresh is not None:
        kwargs["prob_thresh"] = cfg.prob_thresh
    if cfg.nms_thresh is not None:
        kwargs["nms_thresh"] = cfg.nms_thresh
    labels, _details = model.predict_instances(image, **kwargs)
    return np.asarray(labels, dtype=np.int32)


def predict_one(model, image: np.ndarray, cfg: PredictionConfig) -> np.ndarray:
    """Run one model without any additional image normalization."""
    if cfg.family == "cellpose":
        labels = _cellpose_predict(model, image, cfg)
    elif cfg.family == "omnipose":
        labels = _omnipose_predict(model, image, cfg)
    elif cfg.family == "stardist":
        labels = _stardist_predict(model, image, cfg)
    else:
        raise ValueError(cfg.family)
    labels = np.squeeze(labels)
    if labels.ndim != 2 or tuple(labels.shape) != tuple(image.shape[:2]):
        raise ValueError(
            f"Prediction shape {labels.shape} does not match input YX {image.shape[:2]}"
        )
    if np.any(labels < 0):
        raise ValueError("Prediction contains negative labels")
    return labels.astype(np.int32, copy=False)


def _save_prediction(labels: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_label = int(np.max(labels)) if labels.size else 0
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    labels_to_save = labels.astype(dtype, copy=False)
    tiff_path = output_dir / "labels.tif"
    tiff.imwrite(tiff_path, labels_to_save, photometric="minisblack")
    zarr_path = save_ome_zarr(
        output_dir / "labels.ome.zarr",
        labels_to_save,
        "yx",
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        extra_attrs={
            "pft_processing": {
                "operation": "instance_segmentation_prediction",
                "stored_labels": "unique_positive_integer_per_instance",
                "background_label": 0,
            }
        },
    )
    return tiff_path, zarr_path


def run_prediction_dataset(cfg: PredictionConfig) -> Path:
    """Run one model family on all or selected prepared inputs."""
    family, dataset, source_mode = _validate_family_dataset(
        cfg.family, cfg.dataset, cfg.source_mode
    )
    cfg.family, cfg.dataset, cfg.source_mode = family, dataset, source_mode
    items = list_prepared_inputs(cfg.project_root, dataset, source_mode)
    if cfg.selected_sample_keys is not None:
        items = [item for item in items if item.sample_key in cfg.selected_sample_keys]
    if not items:
        raise RuntimeError("No prepared segmentation inputs were selected")

    model = _load_prediction_model(cfg)
    model_name = _model_display_name(cfg)
    output_root = segmentation_prediction_root(
        cfg.project_root, family, dataset, source_mode, model_name
    )
    rows: list[dict[str, Any]] = []
    for item in items:
        image = load_prepared_image(item.input_zarr)
        labels = predict_one(model, image, cfg)
        sample_dir = output_root / Path(item.sample_key)
        tiff_path, zarr_path = _save_prediction(labels, sample_dir)
        rows.append(
            {
                "sample_key": item.sample_key,
                "input_zarr": str(item.input_zarr),
                "labels_tif": str(tiff_path),
                "labels_zarr": str(zarr_path),
                "input_shape": list(image.shape),
                "n_instances": int(np.unique(labels[labels > 0]).size),
                "normalization_inside_model": False,
            }
        )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(cfg),
        "model_name": model_name,
        "rows": rows,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return output_root


def _split_train_validation(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names: list[str],
    fraction: float,
    seed: int,
) -> tuple[tuple[list[np.ndarray], list[np.ndarray], list[str]], tuple[list[np.ndarray], list[np.ndarray], list[str]]]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0,1)")
    indices = list(range(len(images)))
    random.Random(seed).shuffle(indices)
    n_validation = max(1, int(round(len(indices) * fraction))) if len(indices) > 1 and fraction > 0 else 0
    validation_indices = set(indices[:n_validation])
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    train_names: list[str] = []
    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    val_names: list[str] = []
    for index, (image, mask, name) in enumerate(zip(images, masks, names)):
        target = (val_x, val_y, val_names) if index in validation_indices else (train_x, train_y, train_names)
        target[0].append(image)
        target[1].append(mask)
        target[2].append(name)
    if not train_x:
        raise RuntimeError("Training split is empty")
    return (train_x, train_y, train_names), (val_x, val_y, val_names)


def _train_cellpose(cfg: TrainingConfig, run_dir: Path) -> dict[str, Any]:
    from cellpose import models, train

    images, masks, names = collect_training_data(
        cfg.project_root, cfg.dataset, cfg.source_mode
    )
    (train_x, train_y, train_names), (val_x, val_y, val_names) = _split_train_validation(
        images, masks, names, cfg.validation_fraction, cfg.seed
    )
    pretrained = str(cfg.pretrained_model or _select_cellpose_builtin(models))
    model = models.CellposeModel(gpu=cfg.gpu, pretrained_model=pretrained)
    channel_axis = -1 if train_x[0].ndim == 3 else None
    result = train.train_seg(
        model.net,
        train_data=train_x,
        train_labels=train_y,
        test_data=val_x or None,
        test_labels=val_y or None,
        channel_axis=channel_axis,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        normalize=False,
        compute_flows=False,
        save_path=str(run_dir),
        save_every=cfg.save_every,
        min_train_masks=cfg.min_train_masks,
        model_name=cfg.run_name,
    )
    return {
        "pretrained_model": pretrained,
        "train_samples": train_names,
        "validation_samples": val_names,
        "train_result": str(result),
        "normalization_inside_training": False,
    }


def _prepare_omnipose_folder(
    run_dir: Path,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names: list[str],
) -> Path:
    data_dir = run_dir / "prepared_training_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for image, mask, name in zip(images, masks, names):
        stem = _safe_name(name)
        tiff.imwrite(data_dir / f"{stem}_img.tif", image.astype(np.float32))
        tiff.imwrite(data_dir / f"{stem}_masks.tif", mask.astype(np.int32))
    return data_dir


def _train_omnipose(cfg: TrainingConfig, run_dir: Path) -> dict[str, Any]:
    """Train Omnipose through its Python API with normalization disabled."""
    from cellpose_omni import models

    images, masks, names = collect_training_data(
        cfg.project_root, cfg.dataset, cfg.source_mode
    )
    (train_x, train_y, train_names), (val_x, val_y, val_names) = _split_train_validation(
        images, masks, names, cfg.validation_fraction, cfg.seed
    )
    data_dir = _prepare_omnipose_folder(run_dir, images, masks, names)
    nchan = 1 if images[0].ndim == 2 else int(images[0].shape[-1])
    pretrained = str(cfg.pretrained_model or "bact_fluor_omni")
    pretrained_path = Path(pretrained).expanduser()

    constructor_values: dict[str, Any] = {
        "gpu": cfg.gpu,
        "nchan": nchan,
        "nclasses": cfg.omnipose_nclasses,
        "dim": 2,
        "omni": True,
    }
    if pretrained_path.exists():
        constructor_values["pretrained_model"] = str(pretrained_path.resolve())
    elif pretrained.lower() not in {"none", "false", "scratch"}:
        constructor_values["model_type"] = pretrained
    else:
        constructor_values["pretrained_model"] = False

    model = models.CellposeModel(
        **_supported_kwargs(models.CellposeModel, constructor_values)
    )
    train_values: dict[str, Any] = {
        "train_data": train_x,
        "train_labels": train_y,
        "test_data": val_x or None,
        "test_labels": val_y or None,
        "channels": None,
        "channel_axis": -1 if train_x[0].ndim == 3 else None,
        "normalize": False,
        "save_path": str(run_dir),
        "save_every": cfg.save_every,
        "learning_rate": cfg.learning_rate,
        "n_epochs": cfg.epochs,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "rescale": False,
        "min_train_masks": cfg.min_train_masks,
        "netstr": cfg.run_name,
    }
    supported_train_values = _supported_kwargs(model.train, train_values)
    if "normalize" not in supported_train_values:
        raise RuntimeError(
            "The installed Omnipose version does not expose a training normalization "
            "switch. Upgrade Omnipose or use a compatible environment; training was "
            "stopped to prevent unintended second normalization."
        )
    model_path = model.train(**supported_train_values)
    return {
        "pretrained_model": pretrained,
        "train_samples": train_names,
        "validation_samples": val_names,
        "prepared_training_data": str(data_dir),
        "model_path": str(model_path),
        "normalization_inside_training": False,
        "omnipose_api_kwargs": {
            key: value
            for key, value in supported_train_values.items()
            if key not in {"train_data", "train_labels", "test_data", "test_labels"}
        },
    }


def _train_stardist(cfg: TrainingConfig, run_dir: Path) -> dict[str, Any]:
    from stardist.models import Config2D, StarDist2D

    images, masks, names = collect_training_data(
        cfg.project_root, cfg.dataset, cfg.source_mode
    )
    (train_x, train_y, train_names), (val_x, val_y, val_names) = _split_train_validation(
        images, masks, names, cfg.validation_fraction, cfg.seed
    )
    n_channel_in = 1 if train_x[0].ndim == 2 else int(train_x[0].shape[-1])
    config = Config2D(
        n_rays=cfg.n_rays,
        grid=(cfg.grid, cfg.grid),
        train_epochs=cfg.epochs,
        train_steps_per_epoch=cfg.steps_per_epoch,
        train_batch_size=cfg.batch_size,
        train_patch_size=cfg.patch_size,
        train_learning_rate=cfg.learning_rate,
        n_channel_in=n_channel_in,
        use_gpu=cfg.gpu,
    )
    model = StarDist2D(config, name=run_dir.name, basedir=str(run_dir.parent))
    model.train(train_x, train_y, validation_data=(val_x, val_y) if val_x else None)
    if val_x:
        model.optimize_thresholds(val_x, val_y)
    return {
        "train_samples": train_names,
        "validation_samples": val_names,
        "n_channel_in": n_channel_in,
        "normalization_inside_training": False,
    }


def train_initial_model(cfg: TrainingConfig) -> Path:
    """Train or fine-tune one initial model and save it under ``models``."""
    family, dataset, source_mode = _validate_family_dataset(
        cfg.family, cfg.dataset, cfg.source_mode
    )
    cfg.family, cfg.dataset, cfg.source_mode = family, dataset, source_mode
    family_root = segmentation_model_root(
        cfg.project_root, family, dataset, source_mode
    )
    run_dir = family_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if family == "cellpose":
        details = _train_cellpose(cfg, run_dir)
    elif family == "omnipose":
        details = _train_omnipose(cfg, run_dir)
    else:
        details = _train_stardist(cfg, run_dir)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(cfg),
        "details": details,
        "run_dir": str(run_dir),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return run_dir


def semantic_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = np.asarray(reference) > 0
    pred = np.asarray(prediction) > 0
    denominator = int(np.count_nonzero(ref) + np.count_nonzero(pred))
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(ref & pred) / denominator)


def semantic_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = np.asarray(reference) > 0
    pred = np.asarray(prediction) > 0
    union = int(np.count_nonzero(ref | pred))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(ref & pred) / union)


def _instance_iou_matrix(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref_ids = np.unique(reference[reference > 0])
    pred_ids = np.unique(prediction[prediction > 0])
    matrix = np.zeros((len(ref_ids), len(pred_ids)), dtype=np.float64)
    for i, ref_id in enumerate(ref_ids):
        ref_mask = reference == ref_id
        for j, pred_id in enumerate(pred_ids):
            pred_mask = prediction == pred_id
            intersection = np.count_nonzero(ref_mask & pred_mask)
            if intersection == 0:
                continue
            union = np.count_nonzero(ref_mask | pred_mask)
            matrix[i, j] = intersection / union
    return matrix


def instance_f1(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> tuple[float, int, int, int]:
    """Calculate one-to-one instance F1 at the selected IoU threshold."""
    ref_count = int(np.unique(reference[reference > 0]).size)
    pred_count = int(np.unique(prediction[prediction > 0]).size)
    if ref_count == 0 and pred_count == 0:
        return 1.0, 0, 0, 0
    matrix = _instance_iou_matrix(reference, prediction)
    matches = 0
    if matrix.size:
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-matrix)
            matches = sum(matrix[row, col] >= iou_threshold for row, col in zip(rows, cols))
        except Exception:
            candidates = sorted(
                (
                    (matrix[row, col], row, col)
                    for row in range(matrix.shape[0])
                    for col in range(matrix.shape[1])
                    if matrix[row, col] >= iou_threshold
                ),
                reverse=True,
            )
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            for _value, row, col in candidates:
                if row not in used_rows and col not in used_cols:
                    used_rows.add(row)
                    used_cols.add(col)
                    matches += 1
    false_positive = pred_count - matches
    false_negative = ref_count - matches
    denominator = 2 * matches + false_positive + false_negative
    score = 1.0 if denominator == 0 else float(2 * matches / denominator)
    return score, matches, false_positive, false_negative


def _parameter_grid(cfg: TuneConfig) -> list[dict[str, Any]]:
    if cfg.family == "cellpose":
        return [
            {
                "diameter": diameter,
                "flow_threshold": flow,
                "cellprob_threshold": cellprob,
            }
            for diameter, flow, cellprob in product(
                cfg.diameter_values,
                cfg.flow_threshold_values,
                cfg.cellprob_threshold_values,
            )
        ]
    if cfg.family == "omnipose":
        return [
            {
                "diameter": diameter,
                "flow_threshold": flow,
                "mask_threshold": mask_threshold,
            }
            for diameter, flow, mask_threshold in product(
                cfg.diameter_values,
                cfg.flow_threshold_values,
                cfg.mask_threshold_values,
            )
        ]
    return [
        {"prob_thresh": prob, "nms_thresh": nms}
        for prob, nms in product(cfg.prob_thresh_values, cfg.nms_thresh_values)
    ]


def tune_model_parameters(cfg: TuneConfig) -> Path:
    """Evaluate a parameter grid against manual masks and save the best setting."""
    family, dataset, source_mode = _validate_family_dataset(
        cfg.family, cfg.dataset, cfg.source_mode
    )
    cfg.family, cfg.dataset, cfg.source_mode = family, dataset, source_mode
    images, masks, names = collect_training_data(
        cfg.project_root,
        dataset,
        source_mode,
        selected_sample_keys=cfg.selected_sample_keys,
    )
    base_prediction = PredictionConfig(
        project_root=cfg.project_root,
        family=family,
        dataset=dataset,
        source_mode=source_mode,
        model=cfg.model,
        model_name=cfg.model_name,
        gpu=cfg.gpu,
        min_size=cfg.min_size,
        batch_size=cfg.batch_size,
    )
    model = _load_prediction_model(base_prediction)
    rows: list[dict[str, Any]] = []
    for parameter_index, parameters in enumerate(_parameter_grid(cfg), start=1):
        scores = []
        for image, reference, sample_name in zip(images, masks, names):
            prediction_cfg = PredictionConfig(**asdict(base_prediction))
            for key, value in parameters.items():
                setattr(prediction_cfg, key, value)
            prediction = predict_one(model, image, prediction_cfg)
            f1, tp, fp, fn = instance_f1(
                reference,
                prediction,
                iou_threshold=cfg.instance_iou_threshold,
            )
            sample_row = {
                "parameter_index": parameter_index,
                "sample_key": sample_name,
                **parameters,
                "semantic_dice": semantic_dice(reference, prediction),
                "semantic_iou": semantic_iou(reference, prediction),
                "instance_f1": f1,
                "true_positive_instances": tp,
                "false_positive_instances": fp,
                "false_negative_instances": fn,
            }
            scores.append(sample_row)
        aggregate = {
            "parameter_index": parameter_index,
            "sample_key": "__MEAN__",
            **parameters,
            "semantic_dice": float(np.mean([row["semantic_dice"] for row in scores])),
            "semantic_iou": float(np.mean([row["semantic_iou"] for row in scores])),
            "instance_f1": float(np.mean([row["instance_f1"] for row in scores])),
            "true_positive_instances": int(sum(row["true_positive_instances"] for row in scores)),
            "false_positive_instances": int(sum(row["false_positive_instances"] for row in scores)),
            "false_negative_instances": int(sum(row["false_negative_instances"] for row in scores)),
        }
        rows.extend(scores)
        rows.append(aggregate)

    aggregate_rows = [row for row in rows if row["sample_key"] == "__MEAN__"]
    best = max(
        aggregate_rows,
        key=lambda row: (row["instance_f1"], row["semantic_dice"], row["semantic_iou"]),
    )
    model_name = _model_display_name(base_prediction)
    output_root = (
        segmentation_model_root(cfg.project_root, family, dataset, source_mode)
        / "tuning"
        / model_name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "parameter_tuning.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(cfg),
        "best_parameters_and_metrics": best,
        "evaluated_samples": names,
        "normalization_inside_model": False,
        "csv": str(csv_path),
    }
    (output_root / "best_parameters.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return output_root


__all__ = [
    "MODEL_FAMILIES",
    "PredictionConfig",
    "PreparedInputItem",
    "TrainingConfig",
    "TuneConfig",
    "collect_training_data",
    "instance_f1",
    "list_prepared_inputs",
    "load_instance_mask",
    "load_prepared_image",
    "predict_one",
    "run_prediction_dataset",
    "semantic_dice",
    "semantic_iou",
    "train_initial_model",
    "tune_model_parameters",
]
