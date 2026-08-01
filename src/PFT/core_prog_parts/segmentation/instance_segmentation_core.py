"""Run, train, and tune 2D instance-segmentation models on prepared PFT inputs.

All model families consume only ``segmentation_input.ome.zarr`` arrays created
by :mod:`segmentation_input_core`. These arrays are already float32 in [0, 1].
Model-side percentile normalization is therefore disabled to avoid the sparse
zero-background normalization failure documented for U-Net-masked data.

Canonical paths
---------------
Prepared inputs::

    results/segmentation_inputs/<dataset>/<source_mode>/<sample>/
        segmentation_input.ome.zarr

Manual instance masks::

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/mask.tif

Crop-based fine-tuning pairs::

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/
        crops/<train|validation>/<crop_id>/
            segmentation_input.ome.zarr
            mask.tif

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
import gc
import json
import inspect
import hashlib
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
ANNOTATION_SOURCES: tuple[str, ...] = ("crops-only", "full-images-only", "all")
ANNOTATION_SPLITS: tuple[str, ...] = ("train", "validation", "any")
VALIDATION_POLICIES: tuple[str, ...] = ("combined", "legacy-explicit-first")


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
    validation_policy: str = "combined"
    seed: int = 1337
    save_every: int = 50
    min_train_masks: int = 1
    n_rays: int = 32
    grid: int = 2
    steps_per_epoch: int = 100
    patch_size: tuple[int, int] = (256, 256)
    omnipose_nclasses: int = 3
    annotation_source: str = "crops-only"
    require_validation: bool = True
    validation_diameter: float | None = None
    validation_flow_threshold: float = 0.6
    validation_cellprob_threshold: float = -0.5
    validation_mask_threshold: float = 0.0
    validation_min_size: int = 15
    save_validation_labels: bool = True
    evaluate_pretrained_baseline: bool = True
    save_improvement_graphs: bool = True
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
    annotation_source: str = "crops-only"
    annotation_split: str = "validation"
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


def _annotation_kind(sample_key: str) -> str:
    return "crop" if "/crops/" in sample_key.replace("\\", "/") else "full_image"


def _annotation_split(sample_key: str) -> str | None:
    normalized = sample_key.replace("\\", "/")
    if "/crops/train/" in normalized:
        return "train"
    if "/crops/validation/" in normalized:
        return "validation"
    return None


def _source_sample_key(sample_key: str) -> str:
    normalized = sample_key.replace("\\", "/")
    return normalized.split("/crops/", 1)[0]


def list_training_inputs(
    project_root: Path,
    dataset: str,
    source_mode: str,
    *,
    annotation_source: str = "all",
    annotation_split: str = "any",
) -> list[PreparedInputItem]:
    """List full-image and crop-based image/mask pairs for fine-tuning.

    Full-image annotations retain their original sample key. Crop pairs use a
    key of the form ``<sample>/crops/<train|validation>/<crop_id>``. The split
    encoded in this key is respected by the train/validation splitter.
    """
    items = list_prepared_inputs(project_root, dataset, source_mode)
    mask_root = segmentation_mask_root(project_root, dataset, source_mode)
    crop_inputs = sorted(
        mask_root.glob("**/crops/*/crop_*/segmentation_input.ome.zarr")
    )
    for input_zarr in crop_inputs:
        crop_dir = input_zarr.parent
        try:
            relative = crop_dir.relative_to(mask_root)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) < 4 or parts[-3] != "crops" or parts[-2] not in {"train", "validation"}:
            continue
        items.append(
            PreparedInputItem(
                sample_key=relative.as_posix(),
                input_zarr=input_zarr,
                training_mask=crop_dir / "mask.tif",
            )
        )
    if annotation_source not in ANNOTATION_SOURCES:
        raise ValueError(
            f"annotation_source must be one of {ANNOTATION_SOURCES}, received {annotation_source!r}"
        )
    if annotation_split not in ANNOTATION_SPLITS:
        raise ValueError(
            f"annotation_split must be one of {ANNOTATION_SPLITS}, received {annotation_split!r}"
        )
    if annotation_source == "crops-only":
        items = [item for item in items if _annotation_kind(item.sample_key) == "crop"]
    elif annotation_source == "full-images-only":
        items = [item for item in items if _annotation_kind(item.sample_key) == "full_image"]
    if annotation_split != "any":
        items = [
            item for item in items
            if _annotation_split(item.sample_key) == annotation_split
        ]
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
    annotation_source: str = "all",
    annotation_split: str = "any",
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load selected full-image or crop-based pairs with strict checks."""
    items = list_training_inputs(
        project_root,
        dataset,
        source_mode,
        annotation_source=annotation_source,
        annotation_split=annotation_split,
    )
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


def _compact_artifact_name(value: str, index: int, prefix_length: int = 20) -> str:
    """Create a stable short folder name for Windows-safe artifact paths.

    The original sample key is retained in CSV/JSON reports. The short folder
    name prevents failures near the legacy Windows MAX_PATH limit when tuning
    outputs are nested below long project and model directories.
    """
    safe = _safe_name(value).strip(" ._") or "sample"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    prefix = safe[:prefix_length].rstrip(" ._") or "sample"
    return f"s{index:03d}_{prefix}_{digest}"


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
                    nchan=2,
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
                nchan=2,
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



def load_prediction_model(cfg: PredictionConfig):
    """Load one pretrained or user-supplied prediction model for reuse."""
    return _load_prediction_model(cfg)

def _prepare_omnipose_image(image: np.ndarray, dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a two-channel YXC image compatible with bact_fluor_omni.

    The named ``bact_fluor_omni`` weights use ``nchan=2`` and ``nclasses=3``.
    For ``2d_time``, HADA is placed in channel 0 and channel 1 is zero-filled.
    For ``2d_wga_dapi``, WGA is placed first as the cell-boundary signal and
    DAPI second. Other datasets retain their first two numerical channels.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        first = array
        second = np.zeros_like(first, dtype=np.float32)
        policy = "channel0_plus_zero"
    elif array.ndim == 3 and array.shape[-1] >= 1:
        if dataset == "2d_wga_dapi" and array.shape[-1] >= 2:
            first = array[..., 1]  # WGA primary
            second = array[..., 0]  # DAPI secondary
            policy = "wga_then_dapi"
        elif array.shape[-1] == 1:
            first = array[..., 0]
            second = np.zeros_like(first, dtype=np.float32)
            policy = "channel0_plus_zero"
        else:
            first = array[..., 0]
            second = array[..., 1]
            policy = "first_two_channels"
    else:
        raise ValueError(f"Unsupported Omnipose image shape: {array.shape}")
    prepared = np.stack([first, second], axis=-1).astype(np.float32, copy=False)
    return prepared, {
        "policy": policy,
        "output_channels": 2,
        "output_shape": [int(value) for value in prepared.shape],
    }


def _prepare_family_images(
    images: Sequence[np.ndarray], family: str, dataset: str
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if family != "omnipose":
        return [np.asarray(image) for image in images], [
            {"policy": "unchanged", "output_shape": list(np.asarray(image).shape)}
            for image in images
        ]
    prepared: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for image in images:
        converted, record = _prepare_omnipose_image(image, dataset)
        prepared.append(converted)
        records.append(record)
    return prepared, records


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
    prepared, _channel_record = _prepare_omnipose_image(image, cfg.dataset)
    kwargs = {
        "diameter": cfg.diameter,
        "channels": None,
        "channel_axis": -1,
        "omni": True,
        "mask_threshold": cfg.mask_threshold,
        "flow_threshold": cfg.flow_threshold,
        "min_size": cfg.min_size,
        "normalize": False,
    }
    result = model.eval([prepared], **_supported_kwargs(model.eval, kwargs))
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
    validation_policy: str = "combined",
) -> tuple[
    tuple[list[np.ndarray], list[np.ndarray], list[str]],
    tuple[list[np.ndarray], list[np.ndarray], list[str]],
    dict[str, Any],
]:
    """Create a source-aware train/validation split.

    ``combined`` is the default thesis workflow. It preserves explicit
    ``crops/train`` and ``crops/validation`` assignments and additionally
    applies ``validation_fraction`` to source images without an explicit
    assignment. All pairs from one original source image are kept in the same
    split, preventing source-image leakage.

    ``legacy-explicit-first`` reproduces the previous behavior: when at least
    one explicit validation crop exists, all unspecified annotations are put
    into training.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0,1)")
    if validation_policy not in VALIDATION_POLICIES:
        raise ValueError(
            f"validation_policy must be one of {VALIDATION_POLICIES}, "
            f"received {validation_policy!r}"
        )
    if not (len(images) == len(masks) == len(names)):
        raise ValueError("Images, masks, and names must have equal length")

    source_to_indices: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        source_to_indices.setdefault(_source_sample_key(name), []).append(index)

    explicit_train_sources: set[str] = set()
    explicit_validation_sources: set[str] = set()
    for name in names:
        split = _annotation_split(name)
        source = _source_sample_key(name)
        if split == "train":
            explicit_train_sources.add(source)
        elif split == "validation":
            explicit_validation_sources.add(source)

    conflicting_sources = sorted(
        explicit_train_sources & explicit_validation_sources
    )
    if conflicting_sources:
        raise RuntimeError(
            "The following source images contain both explicit training and "
            "validation crops, which would create leakage: "
            + ", ".join(conflicting_sources)
        )

    train_sources = set(explicit_train_sources)
    validation_sources = set(explicit_validation_sources)
    unspecified_sources = sorted(
        set(source_to_indices) - train_sources - validation_sources
    )

    fractional_validation_sources: set[str] = set()
    fractional_train_sources: set[str] = set()
    if validation_policy == "legacy-explicit-first" and validation_sources:
        fractional_train_sources.update(unspecified_sources)
        split_policy = "legacy_explicit_validation_only"
    else:
        shuffled_sources = list(unspecified_sources)
        random.Random(seed).shuffle(shuffled_sources)
        n_validation_sources = (
            max(1, int(round(len(shuffled_sources) * fraction)))
            if len(shuffled_sources) > 1 and fraction > 0
            else 0
        )
        fractional_validation_sources.update(
            shuffled_sources[:n_validation_sources]
        )
        fractional_train_sources.update(
            shuffled_sources[n_validation_sources:]
        )
        split_policy = "combined_explicit_plus_fractional_source_split"

    train_sources.update(fractional_train_sources)
    validation_sources.update(fractional_validation_sources)

    train_indices = {
        index
        for source in train_sources
        for index in source_to_indices[source]
    }
    validation_indices = {
        index
        for source in validation_sources
        for index in source_to_indices[source]
    }
    overlap_indices = train_indices & validation_indices
    if overlap_indices:
        raise RuntimeError(
            "Internal split error: one or more pairs were assigned to both splits"
        )

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    train_names: list[str] = []
    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    val_names: list[str] = []
    split_rows: list[dict[str, Any]] = []

    for index, (image, mask, name) in enumerate(zip(images, masks, names)):
        source = _source_sample_key(name)
        explicit = _annotation_split(name)
        if index in validation_indices:
            final_split = "validation"
            val_x.append(image)
            val_y.append(mask)
            val_names.append(name)
        elif index in train_indices:
            final_split = "train"
            train_x.append(image)
            train_y.append(mask)
            train_names.append(name)
        else:
            raise RuntimeError(f"Pair was not assigned to a split: {name}")

        if source in explicit_validation_sources:
            reason = "explicit_validation_source"
        elif source in explicit_train_sources:
            reason = "explicit_train_source"
        elif source in fractional_validation_sources:
            reason = "fractional_validation_source"
        else:
            reason = "fractional_train_source"
        split_rows.append(
            {
                "sample_key": name,
                "source_sample_key": source,
                "annotation_kind": _annotation_kind(name),
                "explicit_pair_assignment": explicit or "unspecified",
                "final_split": final_split,
                "assignment_reason": reason,
            }
        )

    if not train_x:
        raise RuntimeError("Training split is empty")
    train_source_set = {_source_sample_key(name) for name in train_names}
    validation_source_set = {_source_sample_key(name) for name in val_names}
    leakage_sources = sorted(train_source_set & validation_source_set)
    if leakage_sources:
        raise RuntimeError(
            "Source-image leakage detected after splitting: "
            + ", ".join(leakage_sources)
        )

    diagnostics = {
        "split_policy": split_policy,
        "requested_validation_fraction": fraction,
        "seed": seed,
        "train_pair_count": len(train_names),
        "validation_pair_count": len(val_names),
        "train_source_count": len(train_source_set),
        "validation_source_count": len(validation_source_set),
        "explicit_train_sources": sorted(explicit_train_sources),
        "explicit_validation_sources": sorted(explicit_validation_sources),
        "fractional_train_sources": sorted(fractional_train_sources),
        "fractional_validation_sources": sorted(fractional_validation_sources),
        "train_source_samples": sorted(train_source_set),
        "validation_source_samples": sorted(validation_source_set),
        "source_sample_overlap": [],
        "source_sample_overlap_warning": False,
        "split_rows": split_rows,
    }
    return (train_x, train_y, train_names), (val_x, val_y, val_names), diagnostics


def _save_split_manifest(run_dir: Path, diagnostics: dict[str, Any]) -> Path | None:
    rows = diagnostics.get("split_rows", [])
    if not rows:
        return None
    path = run_dir / "split_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path

def _numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    if array.size == 0 or not np.all(np.isfinite(array)):
        return []
    return [float(item) for item in array]


def _extract_training_result(
    result: Any, run_dir: Path, run_name: str
) -> tuple[Path | None, list[float], list[float], str]:
    model_path: Path | None = None
    train_losses: list[float] = []
    validation_losses: list[float] = []
    if isinstance(result, (tuple, list)):
        if len(result) >= 1 and result[0] is not None:
            candidate = Path(str(result[0])).expanduser()
            if candidate.exists():
                model_path = candidate.resolve()
            elif not candidate.is_absolute():
                joined = run_dir / candidate
                if joined.exists():
                    model_path = joined.resolve()
        if len(result) >= 2:
            train_losses = _numeric_sequence(result[1])
        if len(result) >= 3:
            validation_losses = _numeric_sequence(result[2])
    elif result is not None:
        candidate = Path(str(result)).expanduser()
        if candidate.exists():
            model_path = candidate.resolve()
        elif not candidate.is_absolute() and (run_dir / candidate).exists():
            model_path = (run_dir / candidate).resolve()

    if model_path is None:
        candidates = [
            path for path in run_dir.rglob("*")
            if path.is_file()
            and run_name.lower() in path.name.lower()
            and path.suffix.lower() not in {".json", ".csv", ".png", ".txt"}
        ]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            model_path = candidates[0].resolve()
    return model_path, train_losses, validation_losses, repr(result)


def _save_loss_outputs(
    run_dir: Path,
    train_losses: Sequence[float],
    validation_losses: Sequence[float],
) -> dict[str, str | None]:
    if not train_losses and not validation_losses:
        return {"loss_csv": None, "loss_curve_png": None}
    csv_path = run_dir / "training_losses.csv"
    count = max(len(train_losses), len(validation_losses))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "training_loss", "validation_loss"]
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "epoch": index + 1,
                    "training_loss": train_losses[index] if index < len(train_losses) else "",
                    "validation_loss": (
                        validation_losses[index] if index < len(validation_losses) else ""
                    ),
                }
            )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    if train_losses:
        axis.plot(range(1, len(train_losses) + 1), train_losses, label="training loss")
    if validation_losses:
        axis.plot(
            range(1, len(validation_losses) + 1),
            validation_losses,
            label="validation loss",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Fine-tuning loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    curve_path = run_dir / "training_loss_curve.png"
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)
    return {"loss_csv": str(csv_path), "loss_curve_png": str(curve_path)}


def _display_rgb(image: np.ndarray, dataset: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        rgb = np.zeros((*array.shape, 3), dtype=np.float32)
        rgb[..., 2] = array
        return np.clip(rgb, 0.0, 1.0)
    rgb = np.zeros((*array.shape[:2], 3), dtype=np.float32)
    if dataset == "2d_wga_dapi" and array.shape[-1] >= 2:
        rgb[..., 2] = array[..., 0]
        rgb[..., 1] = array[..., 1]
    else:
        rgb[..., 2] = array[..., 0]
        if array.shape[-1] > 1:
            rgb[..., 1] = array[..., 1]
        if array.shape[-1] > 2:
            rgb[..., 0] = array[..., 2]
    return np.clip(rgb, 0.0, 1.0)


def _save_validation_comparison(
    image: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    output_png: Path,
    dataset: str,
    dice: float,
    iou: float,
) -> None:
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    rgb = _display_rgb(image, dataset)
    overlay = np.array(rgb, copy=True)
    ref_boundary = find_boundaries(reference > 0, mode="outer")
    pred_boundary = find_boundaries(prediction > 0, mode="outer")
    overlay[ref_boundary] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    overlay[pred_boundary] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Prepared input")
    axes[1].imshow(reference > 0, cmap="gray")
    axes[1].set_title("Manual reference")
    axes[2].imshow(prediction, cmap="nipy_spectral")
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay\nDice={dice:.4f}, IoU={iou:.4f}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def _validation_prediction_config(cfg: TrainingConfig, model_path: Path) -> PredictionConfig:
    return PredictionConfig(
        project_root=cfg.project_root,
        family=cfg.family,
        dataset=cfg.dataset,
        source_mode=cfg.source_mode,
        model=model_path,
        model_name=cfg.run_name,
        gpu=cfg.gpu,
        diameter=cfg.validation_diameter,
        flow_threshold=cfg.validation_flow_threshold,
        cellprob_threshold=cfg.validation_cellprob_threshold,
        mask_threshold=cfg.validation_mask_threshold,
        min_size=cfg.validation_min_size,
        batch_size=cfg.batch_size,
    )


def _release_prediction_model(model: Any) -> None:
    """Release a prediction model before training or loading another model."""
    try:
        del model
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _evaluate_model_on_validation(
    cfg: TrainingConfig,
    output_dir: Path,
    prediction_cfg: PredictionConfig,
    validation_images: list[np.ndarray],
    validation_masks: list[np.ndarray],
    validation_names: list[str],
    *,
    evaluation_label: str,
) -> dict[str, Any]:
    """Evaluate one model on a fixed validation set and save binary metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not validation_images:
        payload = {
            "status": "skipped_no_validation_pairs",
            "evaluation_label": evaluation_label,
            "validation_pair_count": 0,
        }
        (output_dir / "validation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    try:
        model = _load_prediction_model(prediction_cfg)
    except Exception as exc:
        payload = {
            "status": "failed_model_loading",
            "evaluation_label": evaluation_label,
            "model": str(prediction_cfg.model),
            "error": repr(exc),
            "validation_pair_count": len(validation_images),
        }
        (output_dir / "validation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    rows: list[dict[str, Any]] = []
    try:
        for image, reference, name in zip(
            validation_images, validation_masks, validation_names
        ):
            prediction = predict_one(model, image, prediction_cfg)
            dice = semantic_dice(reference, prediction)
            iou = semantic_iou(reference, prediction)
            sample_dir = output_dir / _safe_name(name)
            sample_dir.mkdir(parents=True, exist_ok=True)
            comparison_png = sample_dir / "comparison.png"
            _save_validation_comparison(
                image, reference, prediction, comparison_png, cfg.dataset, dice, iou
            )
            label_path: Path | None = None
            if cfg.save_validation_labels:
                max_label = int(np.max(prediction)) if prediction.size else 0
                dtype = (
                    np.uint16
                    if max_label <= np.iinfo(np.uint16).max
                    else np.uint32
                )
                label_path = sample_dir / "predicted_labels.tif"
                tiff.imwrite(label_path, prediction.astype(dtype, copy=False))
            ref_pixels = int(np.count_nonzero(reference))
            pred_pixels = int(np.count_nonzero(prediction))
            rows.append(
                {
                    "sample_key": name,
                    "semantic_dice": dice,
                    "semantic_iou": iou,
                    "reference_foreground_pixels": ref_pixels,
                    "prediction_foreground_pixels": pred_pixels,
                    "foreground_area_ratio": (
                        float(pred_pixels / ref_pixels) if ref_pixels else float("nan")
                    ),
                    "predicted_instances": int(
                        np.unique(prediction[prediction > 0]).size
                    ),
                    "comparison_png": str(comparison_png),
                    "predicted_labels_tif": str(label_path) if label_path else "",
                }
            )
    finally:
        _release_prediction_model(model)

    aggregate = {
        "sample_key": "__MEAN__",
        "semantic_dice": float(np.mean([row["semantic_dice"] for row in rows])),
        "semantic_iou": float(np.mean([row["semantic_iou"] for row in rows])),
        "reference_foreground_pixels": int(
            sum(row["reference_foreground_pixels"] for row in rows)
        ),
        "prediction_foreground_pixels": int(
            sum(row["prediction_foreground_pixels"] for row in rows)
        ),
        "foreground_area_ratio": float(
            np.nanmean([row["foreground_area_ratio"] for row in rows])
        ),
        "predicted_instances": float(
            np.mean([row["predicted_instances"] for row in rows])
        ),
        "comparison_png": "",
        "predicted_labels_tif": "",
    }
    all_rows = rows + [aggregate]
    csv_path = output_dir / "validation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "status": "completed",
        "evaluation_label": evaluation_label,
        "model": str(prediction_cfg.model or prediction_cfg.model_name or "default"),
        "normalization_inside_model": False,
        "validation_parameters": {
            "diameter": cfg.validation_diameter,
            "flow_threshold": cfg.validation_flow_threshold,
            "cellprob_threshold": cfg.validation_cellprob_threshold,
            "mask_threshold": cfg.validation_mask_threshold,
            "min_size": cfg.validation_min_size,
        },
        "mean_semantic_dice": aggregate["semantic_dice"],
        "mean_semantic_iou": aggregate["semantic_iou"],
        "metrics_csv": str(csv_path),
        "rows": rows,
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def _baseline_prediction_config(
    cfg: TrainingConfig, pretrained_model: str | Path | None
) -> PredictionConfig:
    model_value: str | Path | None = pretrained_model
    model_name: str | None = None
    if cfg.family == "omnipose" and pretrained_model is not None:
        candidate = Path(str(pretrained_model)).expanduser()
        if not candidate.exists():
            model_value = None
            model_name = str(pretrained_model)
    return PredictionConfig(
        project_root=cfg.project_root,
        family=cfg.family,
        dataset=cfg.dataset,
        source_mode=cfg.source_mode,
        model=model_value,
        model_name=model_name,
        gpu=cfg.gpu,
        diameter=cfg.validation_diameter,
        flow_threshold=cfg.validation_flow_threshold,
        cellprob_threshold=cfg.validation_cellprob_threshold,
        mask_threshold=cfg.validation_mask_threshold,
        min_size=cfg.validation_min_size,
        batch_size=cfg.batch_size,
    )


def _evaluate_pretrained_baseline(
    cfg: TrainingConfig,
    run_dir: Path,
    pretrained_model: str | Path | None,
    validation_images: list[np.ndarray],
    validation_masks: list[np.ndarray],
    validation_names: list[str],
) -> dict[str, Any]:
    output_dir = run_dir / "validation" / "baseline"
    if not cfg.evaluate_pretrained_baseline:
        payload = {
            "status": "skipped_by_configuration",
            "evaluation_label": "pretrained_baseline",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload
    prediction_cfg = _baseline_prediction_config(cfg, pretrained_model)
    try:
        return _evaluate_model_on_validation(
            cfg,
            output_dir,
            prediction_cfg,
            validation_images,
            validation_masks,
            validation_names,
            evaluation_label="pretrained_baseline",
        )
    except Exception as exc:
        payload = {
            "status": "failed_baseline_evaluation",
            "evaluation_label": "pretrained_baseline",
            "model": str(pretrained_model),
            "error": repr(exc),
            "validation_pair_count": len(validation_images),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload


def _evaluate_validation_after_training(
    cfg: TrainingConfig,
    run_dir: Path,
    model_path: Path | None,
    validation_images: list[np.ndarray],
    validation_masks: list[np.ndarray],
    validation_names: list[str],
) -> dict[str, Any]:
    output_dir = run_dir / "validation"
    if model_path is None or not model_path.exists():
        payload = {
            "status": "skipped_model_path_unresolved",
            "evaluation_label": "fine_tuned",
            "validation_pair_count": len(validation_images),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload
    prediction_cfg = _validation_prediction_config(cfg, model_path)
    return _evaluate_model_on_validation(
        cfg,
        output_dir,
        prediction_cfg,
        validation_images,
        validation_masks,
        validation_names,
        evaluation_label="fine_tuned",
    )


def _short_metric_label(name: str, maximum: int = 44) -> str:
    normalized = name.replace("/crops/validation/", " / val crop ")
    normalized = normalized.replace("/crops/train/", " / train crop ")
    if len(normalized) <= maximum:
        return normalized
    return "..." + normalized[-(maximum - 3):]


def _save_improvement_outputs(
    run_dir: Path,
    baseline: dict[str, Any],
    fine_tuned: dict[str, Any],
) -> dict[str, Any]:
    """Save paired before/after Dice and IoU tables and graphs."""
    output_dir = run_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline.get("status") != "completed" or fine_tuned.get("status") != "completed":
        payload = {
            "status": "skipped_incomplete_evaluations",
            "baseline_status": baseline.get("status"),
            "fine_tuned_status": fine_tuned.get("status"),
        }
        (output_dir / "improvement_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    baseline_rows = {row["sample_key"]: row for row in baseline.get("rows", [])}
    fine_rows = {row["sample_key"]: row for row in fine_tuned.get("rows", [])}
    common_names = sorted(set(baseline_rows) & set(fine_rows))
    if not common_names:
        payload = {"status": "skipped_no_common_validation_samples"}
        (output_dir / "improvement_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    rows: list[dict[str, Any]] = []
    for name in common_names:
        before = baseline_rows[name]
        after = fine_rows[name]
        rows.append(
            {
                "sample_key": name,
                "baseline_semantic_iou": float(before["semantic_iou"]),
                "fine_tuned_semantic_iou": float(after["semantic_iou"]),
                "delta_semantic_iou": float(
                    after["semantic_iou"] - before["semantic_iou"]
                ),
                "baseline_semantic_dice": float(before["semantic_dice"]),
                "fine_tuned_semantic_dice": float(after["semantic_dice"]),
                "delta_semantic_dice": float(
                    after["semantic_dice"] - before["semantic_dice"]
                ),
            }
        )

    mean_row = {
        "sample_key": "__MEAN__",
        "baseline_semantic_iou": float(
            np.mean([row["baseline_semantic_iou"] for row in rows])
        ),
        "fine_tuned_semantic_iou": float(
            np.mean([row["fine_tuned_semantic_iou"] for row in rows])
        ),
        "delta_semantic_iou": float(
            np.mean([row["delta_semantic_iou"] for row in rows])
        ),
        "baseline_semantic_dice": float(
            np.mean([row["baseline_semantic_dice"] for row in rows])
        ),
        "fine_tuned_semantic_dice": float(
            np.mean([row["fine_tuned_semantic_dice"] for row in rows])
        ),
        "delta_semantic_dice": float(
            np.mean([row["delta_semantic_dice"] for row in rows])
        ),
    }
    csv_path = output_dir / "validation_improvement_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mean_row.keys()))
        writer.writeheader()
        writer.writerows(rows + [mean_row])

    graph_paths: dict[str, str] = {}
    import matplotlib.pyplot as plt

    labels = [_short_metric_label(row["sample_key"]) for row in rows]
    positions = np.arange(len(rows), dtype=np.float64)
    width = 0.38
    figure_height = max(5.0, 0.42 * len(rows) + 2.0)

    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        positions - width / 2,
        [row["baseline_semantic_iou"] for row in rows],
        height=width,
        label="Pretrained",
    )
    axis.barh(
        positions + width / 2,
        [row["fine_tuned_semantic_iou"] for row in rows],
        height=width,
        label="Fine-tuned",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Binary semantic IoU")
    axis.set_title("IoU improvement on the held-out validation set")
    axis.legend()
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    iou_path = output_dir / "iou_improvement_by_sample.png"
    figure.savefig(iou_path, dpi=180)
    plt.close(figure)
    graph_paths["iou_improvement_png"] = str(iou_path)

    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        positions,
        [row["delta_semantic_iou"] for row in rows],
    )
    axis.axvline(0.0, linewidth=1.0)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Fine-tuned IoU minus pretrained IoU")
    axis.set_title("Per-sample IoU change after fine-tuning")
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    delta_path = output_dir / "iou_delta_by_sample.png"
    figure.savefig(delta_path, dpi=180)
    plt.close(figure)
    graph_paths["iou_delta_png"] = str(delta_path)

    figure, axis = plt.subplots(figsize=(7, 5))
    metric_positions = np.arange(2, dtype=np.float64)
    axis.bar(
        metric_positions - width / 2,
        [mean_row["baseline_semantic_dice"], mean_row["baseline_semantic_iou"]],
        width=width,
        label="Pretrained",
    )
    axis.bar(
        metric_positions + width / 2,
        [mean_row["fine_tuned_semantic_dice"], mean_row["fine_tuned_semantic_iou"]],
        width=width,
        label="Fine-tuned",
    )
    axis.set_xticks(metric_positions, ["Dice", "IoU"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean binary semantic metric")
    axis.set_title("Mean validation performance before and after fine-tuning")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    mean_path = output_dir / "mean_dice_iou_improvement.png"
    figure.savefig(mean_path, dpi=180)
    plt.close(figure)
    graph_paths["mean_dice_iou_improvement_png"] = str(mean_path)

    payload = {
        "status": "completed",
        "validation_pair_count": len(rows),
        "baseline_mean_semantic_iou": mean_row["baseline_semantic_iou"],
        "fine_tuned_mean_semantic_iou": mean_row["fine_tuned_semantic_iou"],
        "mean_iou_improvement": mean_row["delta_semantic_iou"],
        "baseline_mean_semantic_dice": mean_row["baseline_semantic_dice"],
        "fine_tuned_mean_semantic_dice": mean_row["fine_tuned_semantic_dice"],
        "mean_dice_improvement": mean_row["delta_semantic_dice"],
        "metrics_csv": str(csv_path),
        **graph_paths,
    }
    (output_dir / "improvement_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload

def _train_cellpose(cfg: TrainingConfig, run_dir: Path) -> dict[str, Any]:
    from cellpose import models, train

    images, masks, names = collect_training_data(
        cfg.project_root,
        cfg.dataset,
        cfg.source_mode,
        annotation_source=cfg.annotation_source,
    )
    (train_x, train_y, train_names), (val_x, val_y, val_names), split_info = (
        _split_train_validation(
            images, masks, names, cfg.validation_fraction, cfg.seed, cfg.validation_policy
        )
    )
    if cfg.require_validation and not val_x:
        raise RuntimeError(
            "No validation pairs were selected. Create crops/validation annotations "
            "or use --allow-no-validation explicitly."
        )
    split_manifest = _save_split_manifest(run_dir, split_info)
    pretrained = str(cfg.pretrained_model or _select_cellpose_builtin(models))
    baseline = _evaluate_pretrained_baseline(
        cfg, run_dir, pretrained, val_x, val_y, val_names
    )
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
    model_path, train_losses, validation_losses, result_repr = _extract_training_result(
        result, run_dir, cfg.run_name
    )
    _release_prediction_model(model)
    loss_outputs = _save_loss_outputs(run_dir, train_losses, validation_losses)
    validation = _evaluate_validation_after_training(
        cfg, run_dir, model_path, val_x, val_y, val_names
    )
    improvement = (
        _save_improvement_outputs(run_dir, baseline, validation)
        if cfg.save_improvement_graphs
        else {"status": "skipped_by_configuration"}
    )
    return {
        "pretrained_model": pretrained,
        "split_manifest_csv": str(split_manifest) if split_manifest else None,
        "pretrained_baseline_validation": baseline,
        "model_path": str(model_path) if model_path else None,
        "annotation_source": cfg.annotation_source,
        "train_samples": train_names,
        "validation_samples": val_names,
        "split_diagnostics": split_info,
        "train_result_repr": result_repr,
        "training_losses": train_losses,
        "validation_losses": validation_losses,
        **loss_outputs,
        "automatic_validation": validation,
        "validation_improvement": improvement,
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
    """Fine-tune Omnipose with explicit two-channel bact_fluor_omni inputs."""
    from cellpose_omni import models

    images, masks, names = collect_training_data(
        cfg.project_root,
        cfg.dataset,
        cfg.source_mode,
        annotation_source=cfg.annotation_source,
    )
    (train_raw, train_y, train_names), (val_raw, val_y, val_names), split_info = (
        _split_train_validation(
            images, masks, names, cfg.validation_fraction, cfg.seed, cfg.validation_policy
        )
    )
    if cfg.require_validation and not val_raw:
        raise RuntimeError(
            "No validation pairs were selected. Create crops/validation annotations "
            "or use --allow-no-validation explicitly."
        )
    train_x, train_channel_records = _prepare_family_images(
        train_raw, "omnipose", cfg.dataset
    )
    val_x, val_channel_records = _prepare_family_images(
        val_raw, "omnipose", cfg.dataset
    )
    all_images = train_x + val_x
    all_masks = train_y + val_y
    all_names = train_names + val_names
    data_dir = _prepare_omnipose_folder(run_dir, all_images, all_masks, all_names)
    split_manifest = _save_split_manifest(run_dir, split_info)
    pretrained = str(cfg.pretrained_model or "bact_fluor_omni")
    baseline = _evaluate_pretrained_baseline(
        cfg, run_dir, pretrained, val_raw, val_y, val_names
    )
    pretrained_path = Path(pretrained).expanduser()

    constructor_values: dict[str, Any] = {
        "gpu": cfg.gpu,
        "nchan": 2,
        "nclasses": 3,
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
    # Omnipose 1.1.4 requires link collections to be iterable even when
    # no linked labels are present. Each item corresponds to one image.
    train_links: list[None] = [None] * len(train_y)
    test_links: list[None] | None = [None] * len(val_y) if val_y else None

    train_values: dict[str, Any] = {
        "train_data": train_x,
        "train_labels": train_y,
        "train_links": train_links,
        "test_data": val_x or None,
        "test_labels": val_y or None,
        "test_links": test_links,
        "channels": None,
        "channel_axis": -1,
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
            "The installed Omnipose version does not expose normalize=False for "
            "training. Training was stopped to avoid unintended second normalization."
        )
    result = model.train(**supported_train_values)
    model_path, train_losses, validation_losses, result_repr = _extract_training_result(
        result, run_dir, cfg.run_name
    )
    _release_prediction_model(model)
    loss_outputs = _save_loss_outputs(run_dir, train_losses, validation_losses)
    validation = _evaluate_validation_after_training(
        cfg, run_dir, model_path, val_raw, val_y, val_names
    )
    improvement = (
        _save_improvement_outputs(run_dir, baseline, validation)
        if cfg.save_improvement_graphs
        else {"status": "skipped_by_configuration"}
    )
    return {
        "pretrained_model": pretrained,
        "split_manifest_csv": str(split_manifest) if split_manifest else None,
        "pretrained_baseline_validation": baseline,
        "model_path": str(model_path) if model_path else None,
        "annotation_source": cfg.annotation_source,
        "train_samples": train_names,
        "validation_samples": val_names,
        "split_diagnostics": split_info,
        "prepared_training_data": str(data_dir),
        "train_channel_policy": train_channel_records,
        "validation_channel_policy": val_channel_records,
        "nchan": 2,
        "nclasses": 3,
        "train_result_repr": result_repr,
        "training_losses": train_losses,
        "validation_losses": validation_losses,
        **loss_outputs,
        "automatic_validation": validation,
        "validation_improvement": improvement,
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
        cfg.project_root,
        cfg.dataset,
        cfg.source_mode,
        annotation_source=cfg.annotation_source,
    )
    (train_x, train_y, train_names), (val_x, val_y, val_names), split_info = (
        _split_train_validation(
            images, masks, names, cfg.validation_fraction, cfg.seed, cfg.validation_policy
        )
    )
    if cfg.require_validation and not val_x:
        raise RuntimeError("No validation pairs were selected")
    split_manifest = _save_split_manifest(run_dir, split_info)
    baseline_model = cfg.pretrained_model or "2D_versatile_fluo"
    baseline = _evaluate_pretrained_baseline(
        cfg, run_dir, baseline_model, val_x, val_y, val_names
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
    history = model.train(
        train_x, train_y, validation_data=(val_x, val_y) if val_x else None
    )
    if val_x:
        model.optimize_thresholds(val_x, val_y)
    model_path = run_dir
    train_losses: list[float] = []
    validation_losses: list[float] = []
    history_dict = getattr(history, "history", {}) if history is not None else {}
    if isinstance(history_dict, dict):
        train_losses = _numeric_sequence(history_dict.get("loss"))
        validation_losses = _numeric_sequence(history_dict.get("val_loss"))
    _release_prediction_model(model)
    loss_outputs = _save_loss_outputs(run_dir, train_losses, validation_losses)
    validation = _evaluate_validation_after_training(
        cfg, run_dir, model_path, val_x, val_y, val_names
    )
    improvement = (
        _save_improvement_outputs(run_dir, baseline, validation)
        if cfg.save_improvement_graphs
        else {"status": "skipped_by_configuration"}
    )
    return {
        "model_path": str(model_path),
        "split_manifest_csv": str(split_manifest) if split_manifest else None,
        "pretrained_baseline_validation": baseline,
        "annotation_source": cfg.annotation_source,
        "train_samples": train_names,
        "validation_samples": val_names,
        "split_diagnostics": split_info,
        "n_channel_in": n_channel_in,
        "training_losses": train_losses,
        "validation_losses": validation_losses,
        **loss_outputs,
        "automatic_validation": validation,
        "validation_improvement": improvement,
        "normalization_inside_training": False,
    }


def train_initial_model(cfg: TrainingConfig) -> Path:
    """Train or fine-tune one model and save structured validation outputs."""
    family, dataset, source_mode = _validate_family_dataset(
        cfg.family, cfg.dataset, cfg.source_mode
    )
    if cfg.annotation_source not in ANNOTATION_SOURCES:
        raise ValueError(f"Invalid annotation_source: {cfg.annotation_source}")
    if cfg.validation_policy not in VALIDATION_POLICIES:
        raise ValueError(f"Invalid validation_policy: {cfg.validation_policy}")
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
    """Tune inference parameters on manual masks using binary Dice and IoU."""
    family, dataset, source_mode = _validate_family_dataset(
        cfg.family, cfg.dataset, cfg.source_mode
    )
    cfg.family, cfg.dataset, cfg.source_mode = family, dataset, source_mode
    images, masks, names = collect_training_data(
        cfg.project_root,
        dataset,
        source_mode,
        selected_sample_keys=cfg.selected_sample_keys,
        annotation_source=cfg.annotation_source,
        annotation_split=cfg.annotation_split,
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
    model_name = _model_display_name(base_prediction)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        segmentation_model_root(cfg.project_root, family, dataset, source_mode)
        / "tuning"
        / model_name
        / timestamp
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for parameter_index, parameters in enumerate(_parameter_grid(cfg), start=1):
        sample_rows: list[dict[str, Any]] = []
        for sample_index, (image, reference, sample_name) in enumerate(
            zip(images, masks, names), start=1
        ):
            prediction_cfg = PredictionConfig(**asdict(base_prediction))
            for key, value in parameters.items():
                setattr(prediction_cfg, key, value)
            prediction = predict_one(model, image, prediction_cfg)
            dice = semantic_dice(reference, prediction)
            iou = semantic_iou(reference, prediction)
            ref_pixels = int(np.count_nonzero(reference))
            pred_pixels = int(np.count_nonzero(prediction))
            parameter_dir = output_root / f"parameter_{parameter_index:03d}"
            artifact_name = _compact_artifact_name(sample_name, sample_index)
            sample_dir = parameter_dir / artifact_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            comparison_png = sample_dir / "comparison.png"
            _save_validation_comparison(
                image, reference, prediction, comparison_png, dataset, dice, iou
            )
            label_path = sample_dir / "predicted_labels.tif"
            max_label = int(np.max(prediction)) if prediction.size else 0
            dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
            tiff.imwrite(label_path, prediction.astype(dtype, copy=False))
            sample_row = {
                "parameter_index": parameter_index,
                "sample_key": sample_name,
                "artifact_name": artifact_name,
                "artifact_directory": str(sample_dir),
                **parameters,
                "semantic_dice": dice,
                "semantic_iou": iou,
                "reference_foreground_pixels": ref_pixels,
                "prediction_foreground_pixels": pred_pixels,
                "foreground_area_ratio": (
                    float(pred_pixels / ref_pixels) if ref_pixels else float("nan")
                ),
                "predicted_instances": int(np.unique(prediction[prediction > 0]).size),
                "comparison_png": str(comparison_png),
                "predicted_labels_tif": str(label_path),
            }
            sample_rows.append(sample_row)
        aggregate = {
            "parameter_index": parameter_index,
            "sample_key": "__MEAN__",
            "artifact_name": "",
            "artifact_directory": "",
            **parameters,
            "semantic_dice": float(np.mean([row["semantic_dice"] for row in sample_rows])),
            "semantic_iou": float(np.mean([row["semantic_iou"] for row in sample_rows])),
            "reference_foreground_pixels": int(
                sum(row["reference_foreground_pixels"] for row in sample_rows)
            ),
            "prediction_foreground_pixels": int(
                sum(row["prediction_foreground_pixels"] for row in sample_rows)
            ),
            "foreground_area_ratio": float(
                np.nanmean([row["foreground_area_ratio"] for row in sample_rows])
            ),
            "predicted_instances": float(
                np.mean([row["predicted_instances"] for row in sample_rows])
            ),
            "comparison_png": "",
            "predicted_labels_tif": "",
        }
        rows.extend(sample_rows)
        rows.append(aggregate)

    aggregate_rows = [row for row in rows if row["sample_key"] == "__MEAN__"]
    best = max(
        aggregate_rows,
        key=lambda row: (row["semantic_iou"], row["semantic_dice"]),
    )
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
        "ranking": "mean semantic IoU, then mean semantic Dice",
        "normalization_inside_model": False,
        "csv": str(csv_path),
    }
    (output_root / "best_parameters.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return output_root


__all__ = [
    "ANNOTATION_SOURCES",
    "ANNOTATION_SPLITS",
    "MODEL_FAMILIES",
    "PredictionConfig",
    "PreparedInputItem",
    "TrainingConfig",
    "TuneConfig",
    "collect_training_data",
    "instance_f1",
    "list_prepared_inputs",
    "list_training_inputs",
    "load_instance_mask",
    "load_prediction_model",
    "load_prepared_image",
    "predict_one",
    "run_prediction_dataset",
    "semantic_dice",
    "semantic_iou",
    "train_initial_model",
    "tune_model_parameters",
]
