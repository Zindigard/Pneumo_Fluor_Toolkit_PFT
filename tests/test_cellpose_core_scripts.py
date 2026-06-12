from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

RUN_CASES = [
    ("PFT.core_prog_parts.cellpose_run_2d_time_core", "CellposeRun2DTimeConfig", "2d_time", (0,)),
    ("PFT.core_prog_parts.cellpose_run_2d_wga_dapi_core", "CellposeRun2DWgaDapiConfig", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.cellpose_run_3d_core", "CellposeRun3DConfig", "3d", (0, 1, 2)),
]

TRAIN_CASES = [
    ("PFT.core_prog_parts.cellpose_finetune_2d_time_core", "CellposeFineTune2DTimeConfig", "2d_time", (0,)),
    ("PFT.core_prog_parts.cellpose_finetune_2d_wga_dapi_core", "CellposeFineTune2DWgaDapiConfig", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.cellpose_finetune_3d_core", "CellposeFineTune3DConfig", "3d", (0, 1, 2)),
]


def _install_fake_cellpose(monkeypatch, eval_factory=None, train_result="trained-model"):
    calls = {"models": [], "eval": [], "train": []}

    class FakeCellposeModel:
        def __init__(self, **kwargs):
            calls["models"].append(kwargs)
            self.net = object()

        def eval(self, img, **kwargs):
            calls["eval"].append((np.asarray(img), kwargs))
            if eval_factory is None:
                shape = np.asarray(img).shape[:2]
                masks = np.zeros(shape, dtype=np.int32)
                masks[1:-1, 1:-1] = 1
            else:
                masks = eval_factory(np.asarray(img))
            return masks, {}, np.zeros(1)

    def fake_train_seg(net, **kwargs):
        calls["train"].append((net, kwargs))
        return train_result

    cellpose = types.ModuleType("cellpose")
    cellpose.models = types.SimpleNamespace(CellposeModel=FakeCellposeModel)
    cellpose.train = types.SimpleNamespace(train_seg=fake_train_seg)
    monkeypatch.setitem(sys.modules, "cellpose", cellpose)
    return calls


@pytest.mark.parametrize("module_name,config_name,dataset,channels", RUN_CASES)
def test_cellpose_run_configs(module_name, config_name, dataset, channels, tmp_path):
    mod = importlib.import_module(module_name)
    cfg = getattr(mod, config_name)(project_root=tmp_path, gpu=False)
    assert cfg.dataset == dataset
    assert cfg.root() == tmp_path
    assert cfg.min_size > 0


@pytest.mark.parametrize("module_name,config_name,dataset,channels", RUN_CASES)
def test_cellpose_run_dataset_with_fake_model(monkeypatch, tmp_path, module_name, config_name, dataset, channels):
    mod = importlib.import_module(module_name)
    calls = _install_fake_cellpose(monkeypatch)

    model_path = tmp_path / "models" / f"cellpose_{dataset}" / "model.pth"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    image_path = tmp_path / "inputs" / "sample_a" / "image.ome.zarr"
    image_path.mkdir(parents=True)

    monkeypatch.setattr(mod, "_list_input_images", lambda root, ds: [image_path])
    saved = []

    def fake_save(out_dir, labels, axes):
        out_dir.mkdir(parents=True, exist_ok=True)
        zarr = out_dir / "labels.ome.zarr"
        zarr.mkdir(exist_ok=True)
        saved.append((out_dir, np.asarray(labels), axes))
        return zarr

    monkeypatch.setattr(mod, "_save_labels_omezarr_and_tiff", fake_save)
    if dataset == "3d":
        volume = np.zeros((3, 12, 14, 3), dtype=np.float32)
        monkeypatch.setattr(mod, "_extract_3d_volume_as_slices", lambda path, channels: volume)
    else:
        image = np.zeros((12, 14, len(channels)), dtype=np.float32) if len(channels) > 1 else np.zeros((12, 14), dtype=np.float32)
        monkeypatch.setattr(mod, "_extract_2d_image", lambda path, channels: image)

    cfg = getattr(mod, config_name)(project_root=tmp_path, model_path=model_path, gpu=False)
    out = mod.run_cellpose_dataset(cfg)

    assert out.is_dir()
    assert len(saved) == 1
    expected_axes = "zyx" if dataset == "3d" else "yx"
    assert saved[0][2] == expected_axes
    expected_shape = (3, 12, 14) if dataset == "3d" else (12, 14)
    assert saved[0][1].shape == expected_shape
    assert len(calls["eval"]) == (3 if dataset == "3d" else 1)
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["sample"] == "sample_a"
    assert summary[0]["labels_shape"] == list(expected_shape)


@pytest.mark.parametrize("module_name,config_name,dataset,channels", TRAIN_CASES)
def test_cellpose_finetune_with_fake_training(monkeypatch, tmp_path, module_name, config_name, dataset, channels):
    mod = importlib.import_module(module_name)
    calls = _install_fake_cellpose(monkeypatch)
    pairs = [(tmp_path / "image.ome.zarr", tmp_path / "mask.tif", "sample_a")]
    monkeypatch.setattr(mod, "_collect_training_pairs", lambda root, ds: pairs)

    if dataset == "3d":
        volume = np.zeros((2, 10, 11, 3), dtype=np.float32)
        masks = np.zeros((2, 10, 11), dtype=np.int32)
        masks[:, 2:8, 3:9] = 1
        monkeypatch.setattr(mod, "_extract_3d_volume_as_slices", lambda path, channels: volume)
        monkeypatch.setattr(mod, "_load_3d_mask", lambda path: masks)
    else:
        image = np.zeros((10, 11, len(channels)), dtype=np.float32) if len(channels) > 1 else np.zeros((10, 11), dtype=np.float32)
        mask = np.zeros((10, 11), dtype=np.int32)
        mask[2:8, 3:9] = 1
        monkeypatch.setattr(mod, "_extract_2d_image", lambda path, channels: image)
        monkeypatch.setattr(mod, "_load_2d_mask", lambda path: mask)

    cfg = getattr(mod, config_name)(project_root=tmp_path, gpu=False, n_epochs=2, batch_size=1)
    out = mod.finetune_cellpose(cfg)

    assert out.is_dir()
    assert len(calls["train"]) == 1
    kwargs = calls["train"][0][1]
    assert kwargs["n_epochs"] == 2
    assert kwargs["batch_size"] == 1
    expected_arrays = 2 if dataset == "3d" else 1
    total_arrays = len(kwargs["train_data"]) + len(kwargs.get("test_data") or [])
    assert total_arrays == expected_arrays
    summary = json.loads((out / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["input_dataset"] == dataset
    assert tuple(summary["image_channels"]) == channels


def test_cellpose_split_train_val_is_deterministic():
    mod = importlib.import_module("PFT.core_prog_parts.cellpose_finetune_2d_time_core")
    images = [np.full((2, 2), i) for i in range(5)]
    labels = [np.full((2, 2), i) for i in range(5)]
    names = [f"s{i}" for i in range(5)]
    a = mod._split_train_val(images, labels, names, 0.4, 7)
    b = mod._split_train_val(images, labels, names, 0.4, 7)
    assert a[0][2] == b[0][2]
    assert a[1][2] == b[1][2]
    assert len(a[1][0]) == 2
