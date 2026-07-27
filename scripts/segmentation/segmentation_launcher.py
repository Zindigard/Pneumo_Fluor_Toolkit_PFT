from __future__ import annotations
# Permit direct execution from a source checkout after scripts were grouped into subpackages.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()
for _pft_parent in [_PFT_SCRIPT_FILE.parent, *_PFT_SCRIPT_FILE.parents]:
    _pft_src = _pft_parent / "src"
    if (_pft_src / "PFT").exists():
        if str(_pft_src) not in _pft_sys.path:
            _pft_sys.path.insert(0, str(_pft_src))
        break

def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Locate the repository root independently of script nesting depth."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "PFT").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return _PFTPath.cwd()


import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


# Make sure imports work when this file is placed in <project>/scripts
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT_GUESS = _pft_project_root(_THIS_FILE)
_SRC = _PROJECT_ROOT_GUESS / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PFT.core_prog_parts.common_paths import find_project_root, normalize_dataset_name
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.segmentation.cellpose_train import CellposeTrainConfig, train_cellpose_model
from PFT.core_prog_parts.segmentation.cellpose_run import CellposeRunConfig, run_cellpose_dataset
from PFT.core_prog_parts.segmentation.stardist_train import StarDistTrainConfig, train_stardist_model
from PFT.core_prog_parts.segmentation.stardist_run import StarDistRunConfig, run_stardist_dataset
from PFT.core_prog_parts.segmentation.omnipose_train import OmniposeTrainConfig, train_omnipose_model
from PFT.core_prog_parts.segmentation.omnipose_run import OmniposeRunConfig, run_omnipose_dataset

"""Interactive launcher for segmentation training and prediction workflows."""


def prompt_choice(title: str, options: list[str], default: int = 0) -> str:
    """Ask the user to choose one option from a numbered list."""
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        tag = " (default)" if i - 1 == default else ""
        print(f"  {i}) {opt}{tag}")
    while True:
        s = input(f"Choose number [1-{len(options)}] or Enter for default: ").strip()
        if s == "":
            return options[default]
        try:
            idx = int(s) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(options):
            return options[idx]
        print("Invalid choice.")


def prompt_bool(title: str, default: bool = True) -> bool:
    """Ask the user a yes or no question."""
    suffix = "Y/n" if default else "y/N"
    while True:
        s = input(f"{title} [{suffix}]: ").strip().lower()
        if s == "":
            return default
        if s in {"y", "yes", "1", "true"}:
            return True
        if s in {"n", "no", "0", "false"}:
            return False
        print("Please answer yes or no.")


def prompt_text(title: str, default: str = "") -> str:
    """Ask the user for a text value."""
    s = input(f"{title} [{default}]: ").strip()
    return s if s else default


def prompt_int(title: str, default: int) -> int:
    """Ask the user for an integer value."""
    while True:
        s = input(f"{title} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


def prompt_float(title: str, default: float) -> float:
    """Ask the user for a float value."""
    while True:
        s = input(f"{title} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            print("Please enter a number.")


def choose_dataset(ndim: int) -> str:
    """Let the user choose a dataset name for 2D or 3D workflows."""
    suggestions = ["2d_time", "2d_wga_dapi"] if ndim == 2 else ["3d_time", "3d_wga_dapi"]
    opts = suggestions + ["custom"]
    choice = prompt_choice("Choose dataset:", opts, default=0)
    if choice == "custom":
        return prompt_text("Enter dataset name", suggestions[0])
    return choice


def default_nchan(dataset: str) -> int:
    """Return the expected number of input channels for one dataset."""
    ds = normalize_dataset_name(dataset)
    return 2 if ds == "2d_wga_dapi" or "wga" in ds else 1


def _candidate_mask_files(sample_dir: Path) -> list[Path]:
    """List likely instance mask files for one sample folder."""
    names = ["mask.tif", "mask.tiff", "masks.tif", "masks.tiff", "labels.tif", "labels.tiff"]
    out: list[Path] = []
    for name in names:
        p = sample_dir / name
        if p.exists():
            out.append(p)
    for ext in ("*.tif", "*.tiff"):
        for p in sorted(sample_dir.glob(ext)):
            if p not in out:
                out.append(p)
    return out


def _center_index(n: int) -> int:
    """Return the center index for one axis."""
    return max(0, int(n) // 2)


def _load_mask_for_metrics(mask_path: Path, ndim: int) -> np.ndarray:
    """Load one ground-truth label mask for metric computation."""
    arr = np.asarray(tiff.imread(mask_path))
    if ndim == 2 and arr.ndim == 3:
        arr = arr[_center_index(arr.shape[0])]
    return arr.astype(np.int32, copy=False)


def compute_binary_metrics(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict[str, float | int]:
    """Compute simple foreground metrics between prediction and ground truth."""
    pred_fg = np.asarray(pred_labels > 0, dtype=bool)
    gt_fg = np.asarray(gt_labels > 0, dtype=bool)

    if pred_fg.shape != gt_fg.shape:
        raise ValueError(f"Prediction and ground-truth shapes differ: {pred_fg.shape} vs {gt_fg.shape}")

    tp = int(np.count_nonzero(pred_fg & gt_fg))
    fp = int(np.count_nonzero(pred_fg & ~gt_fg))
    fn = int(np.count_nonzero(~pred_fg & gt_fg))
    tn = int(np.count_nonzero(~pred_fg & ~gt_fg))

    dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    pred_instances = int(len(np.unique(pred_labels)) - (1 if np.any(pred_labels == 0) else 0))
    gt_instances = int(len(np.unique(gt_labels)) - (1 if np.any(gt_labels == 0) else 0))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice_fg": float(dice),
        "iou_fg": float(iou),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "accuracy_fg": float(accuracy),
        "pred_instances": pred_instances,
        "gt_instances": gt_instances,
        "pred_foreground_px": int(np.count_nonzero(pred_fg)),
        "gt_foreground_px": int(np.count_nonzero(gt_fg)),
    }


def save_labels_as_omezarr(out_dir: Path, labels: np.ndarray, source_path: str = "") -> Path:
    """Save integer label masks as an OME-Zarr result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = "yx" if labels.ndim == 2 else "zyx"
    meta = SimpleNamespace(
        pixel_size_um_x=1.0,
        pixel_size_um_y=1.0,
        pixel_size_um_z=1.0,
        channel_names=[],
        source_path=source_path,
        axes=axes,
    )
    save_ome_zarr_next_to_outputs(
        out_dir,
        np.asarray(labels, dtype=np.int32),
        meta,
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
    )
    return out_dir / "image.ome.zarr"


def _extract_model_name_from_output(out_root: Path) -> str:
    """Infer the model folder name from the raw prediction path."""
    return out_root.name


def postprocess_prediction_outputs(project_root: Path, family: str, dataset: str, ndim: int, raw_out_root: Path) -> Path:
    """Convert prediction TIFF labels into OME-Zarr outputs and compute metrics."""
    model_name = _extract_model_name_from_output(raw_out_root)
    final_root = project_root / "results" / "segmented" / family / normalize_dataset_name(dataset) / model_name
    final_root.mkdir(parents=True, exist_ok=True)

    mask_root = project_root / "results" / "segmentation_masks" / ("2d" if ndim == 2 else "3d") / normalize_dataset_name(dataset)
    rows: list[dict[str, Any]] = []

    for sample_dir in sorted(p for p in raw_out_root.iterdir() if p.is_dir()):
        labels_tif = sample_dir / "labels.tif"
        if not labels_tif.exists():
            continue

        sample = sample_dir.name
        pred = np.asarray(tiff.imread(labels_tif), dtype=np.int32)
        final_sample_dir = final_root / sample
        final_sample_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = save_labels_as_omezarr(final_sample_dir, pred, source_path=str(labels_tif))
        tiff.imwrite(final_sample_dir / "labels.tif", pred.astype(np.int32))

        row: dict[str, Any] = {
            "sample": sample,
            "prediction_tif": str(labels_tif),
            "prediction_zarr": str(zarr_path),
            "shape": list(pred.shape),
            "pred_instances": int(len(np.unique(pred)) - (1 if np.any(pred == 0) else 0)),
        }

        gt_dir = mask_root / sample
        gt_candidates = _candidate_mask_files(gt_dir) if gt_dir.exists() else []
        if gt_candidates:
            gt = _load_mask_for_metrics(gt_candidates[0], ndim)
            try:
                row.update(compute_binary_metrics(pred, gt))
                row["gt_mask_path"] = str(gt_candidates[0])
            except ValueError as exc:
                row["metrics_error"] = str(exc)
        else:
            row["gt_mask_path"] = None

        (final_sample_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    summary_json = final_root / "metrics_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        csv_path = final_root / "metrics_summary.csv"
        keys = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        means: dict[str, float] = {}
        metric_keys = ["dice_fg", "iou_fg", "precision_fg", "recall_fg", "accuracy_fg"]
        for key in metric_keys:
            vals = [float(r[key]) for r in rows if key in r]
            if vals:
                means[key] = float(np.mean(vals))
        (final_root / "metrics_mean.json").write_text(json.dumps(means, indent=2), encoding="utf-8")

    return final_root


def _find_history_arrays(run_dir: Path) -> dict[str, np.ndarray]:
    """Search a model folder for saved loss arrays or JSON histories."""
    found: dict[str, np.ndarray] = {}

    json_candidates = list(run_dir.glob("*.json"))
    for jp in json_candidates:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    found[k] = np.asarray(v, dtype=float)

    npy_candidates = list(run_dir.glob("*.npy")) + list(run_dir.rglob("*.npy"))
    for npy in npy_candidates:
        name = npy.stem.lower()
        try:
            arr = np.asarray(np.load(npy, allow_pickle=False), dtype=float).ravel()
        except Exception:
            continue
        if arr.size < 2:
            continue
        if any(tag in name for tag in ["loss", "train", "test", "val"]):
            found[name] = arr

    return found


def try_create_training_graph(run_dir: Path) -> Path | None:
    """Try to create a training-curve figure from saved history files."""
    series = _find_history_arrays(run_dir)
    if not series:
        return None

    keys = sorted(series.keys())[:6]
    fig = plt.figure(figsize=(10, 5), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    for key in keys:
        y = series[key]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, linewidth=2, label=key)
    ax.set_title(f"Training history: {run_dir.name}")
    ax.set_xlabel("Epoch / checkpoint")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()

    out_png = run_dir / "training_history.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_training(project_root: Path, family: str, dataset: str, ndim: int) -> None:
    """Collect terminal choices and launch one training workflow."""
    gpu = prompt_bool("Use GPU", True)
    default_name = f"{family}_{normalize_dataset_name(dataset)}_{ndim}d"
    model_name = prompt_text("Model name", default_name)

    if family == "cellpose":
        cfg = CellposeTrainConfig(
            project_root=project_root,
            dataset=dataset,
            ndim=ndim,
            pretrained_model=prompt_text("Pretrained model", "cpsam"),
            gpu=gpu,
            learning_rate=prompt_float("Learning rate", 1e-5),
            weight_decay=prompt_float("Weight decay", 0.1),
            batch_size=prompt_int("Batch size", 4),
            n_epochs=prompt_int("Epochs", 200),
            min_train_masks=prompt_int("Minimum train masks", 1),
            test_fraction=prompt_float("Validation fraction", 0.2),
            compute_flows=prompt_bool("Compute flows", False),
            save_every=prompt_int("Save every N epochs", 50),
            model_name=model_name,
        )
        run_dir = train_cellpose_model(cfg)
    elif family == "stardist":
        cfg = StarDistTrainConfig(
            project_root=project_root,
            dataset=dataset,
            ndim=ndim,
            model_name=model_name,
            n_rays=prompt_int("Number of rays", 32),
            grid=prompt_int("Grid", 2),
            epochs=prompt_int("Epochs", 200),
            steps_per_epoch=prompt_int("Steps per epoch", 100),
            train_batch_size=prompt_int("Batch size", 4),
            learning_rate=prompt_float("Learning rate", 3e-4),
            train_fraction=prompt_float("Train fraction", 0.8),
            use_gpu=gpu,
        )
        run_dir = train_stardist_model(cfg)
    else:
        cfg = OmniposeTrainConfig(
            project_root=project_root,
            dataset=dataset,
            ndim=ndim,
            model_name=model_name,
            pretrained_model=prompt_text("Pretrained model", "None"),
            nchan=prompt_int("Number of input channels", default_nchan(dataset)),
            nclasses=prompt_int("Number of classes", 3),
            diameter=prompt_float("Diameter", 0.0),
            learning_rate=prompt_float("Learning rate", 0.1),
            batch_size=prompt_int("Batch size", 8),
            n_epochs=prompt_int("Epochs", 400),
            use_gpu=gpu,
        )
        run_dir = train_omnipose_model(cfg)

    graph = try_create_training_graph(run_dir)
    print(f"\nTraining outputs saved to: {run_dir}")
    if graph is not None:
        print(f"Training history graph saved to: {graph}")
    else:
        print("No plottable training history file was found automatically.")



def run_prediction(project_root: Path, family: str, dataset: str, ndim: int) -> None:
    """Collect terminal choices, run prediction, and standardize saved outputs."""
    gpu = prompt_bool("Use GPU", True)
    if family == "cellpose":
        pretrained_model = prompt_text("Pretrained model or saved model path", "cpsam")
        cfg = CellposeRunConfig(
            project_root=project_root,
            dataset=dataset,
            ndim=ndim,
            pretrained_model=pretrained_model,
            diameter=None,
            flow_threshold=prompt_float("Flow threshold", 0.4),
            cellprob_threshold=prompt_float("Cell probability threshold", 0.0),
            min_size=prompt_int("Minimum object size", 15),
            batch_size=prompt_int("Batch size", 8),
            gpu=gpu,
            model_name=prompt_text("Output model name", Path(pretrained_model).stem if pretrained_model else "cellpose"),
        )
        raw_out = run_cellpose_dataset(cfg)
    elif family == "stardist":
        model_name = prompt_text("Model name", f"stardist_{ndim}d_{normalize_dataset_name(dataset)}")
        cfg = StarDistRunConfig(
            project_root=project_root,
            dataset=dataset,
            ndim=ndim,
            model_name=model_name,
            prob_thresh=None,
            nms_thresh=None,
        )
        raw_out = run_stardist_dataset(cfg)
    else:
        model_name = prompt_text("Output model name", f"omnipose_{normalize_dataset_name(dataset)}_{ndim}d")
        pretrained_model = prompt_text("Custom pretrained model path (leave empty for built-in)", "")
        cfg = OmniposeRunConfig(
            project_root=project_root,
            dataset=dataset,
            ndim=ndim,
            model_name=model_name,
            pretrained_model=pretrained_model or None,
            model_type=prompt_text("Built-in model type", "cyto2_omni"),
            nchan=prompt_int("Number of input channels", default_nchan(dataset)),
            nclasses=prompt_int("Number of classes", 3),
            diameter=None,
            mask_threshold=prompt_float("Mask threshold", 0.0),
            flow_threshold=prompt_float("Flow threshold", 0.4),
            min_size=prompt_int("Minimum object size", 15),
            gpu=gpu,
        )
        raw_out = run_omnipose_dataset(cfg)

    final_out = postprocess_prediction_outputs(project_root, family, dataset, ndim, raw_out)
    print(f"\nRaw prediction folder: {raw_out}")
    print(f"Standardized segmented OME-Zarr outputs: {final_out}")



def main() -> None:
    """Run the interactive segmentation launcher."""
    project_root = find_project_root(_THIS_FILE)
    print(f"Project root: {project_root}")

    mode = prompt_choice("Choose workflow:", ["train / fine-tune", "run / predict"], default=0)
    family = prompt_choice("Choose model family:", ["cellpose", "stardist", "omnipose"], default=0)
    ndim_text = prompt_choice("Choose dimensionality:", ["2D", "3D"], default=0)
    ndim = 2 if ndim_text == "2D" else 3
    dataset = choose_dataset(ndim)

    if mode.startswith("train"):
        run_training(project_root, family, dataset, ndim)
    else:
        run_prediction(project_root, family, dataset, ndim)


if __name__ == "__main__":
    main()
