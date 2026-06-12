from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TRAIN_CASES = [
    ("PFT.core_prog_parts.stardist_finetune_2d_time_core", "finetune_stardist_2d_time", "2d_time", (0,)),
    ("PFT.core_prog_parts.stardist_finetune_2d_wga_dapi_core", "finetune_stardist_2d_wga_dapi", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.stardist_finetune_3d_core", "finetune_stardist_3d", "3d", (0,)),
]
RUN_CASES = [
    ("PFT.core_prog_parts.stardist_run_2d_time_core", "run_stardist_2d_time", "2d_time", (0,)),
    ("PFT.core_prog_parts.stardist_run_2d_wga_dapi_core", "run_stardist_2d_wga_dapi", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.stardist_run_3d_core", "run_stardist_3d", "3d", (0,)),
]


def _install_fake_stardist(monkeypatch):
    calls = {"config": [], "model": [], "train": []}

    class FakeConfig2D:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["config"].append(kwargs)

    class FakeStarDist2D:
        def __init__(self, conf, name, basedir):
            self.conf = conf
            self.name = name
            self.basedir = basedir
            calls["model"].append((conf, name, basedir))

        def train(self, x, y, **kwargs):
            calls["train"].append((x, y, kwargs))

        def predict_instances(self, img, **kwargs):
            shape = np.asarray(img).shape[:2]
            labels = np.zeros(shape, dtype=np.uint16)
            labels[1:-1, 1:-1] = 1
            return labels, {"kwargs": kwargs}

    pkg = types.ModuleType("stardist")
    models = types.ModuleType("stardist.models")
    models.Config2D = FakeConfig2D
    models.StarDist2D = FakeStarDist2D
    pkg.models = models
    monkeypatch.setitem(sys.modules, "stardist", pkg)
    monkeypatch.setitem(sys.modules, "stardist.models", models)
    return calls


@pytest.mark.parametrize("module_name,function_name,dataset,channels", TRAIN_CASES)
def test_stardist_training_arrays_and_finetune(monkeypatch, tmp_path, module_name, function_name, dataset, channels):
    mod = importlib.import_module(module_name)
    calls = _install_fake_stardist(monkeypatch)
    pairs = [
        (tmp_path / "a.ome.zarr", tmp_path / "a.tif", "a"),
        (tmp_path / "b.ome.zarr", tmp_path / "b.tif", "b"),
    ]
    monkeypatch.setattr(mod, "collect_training_pairs", lambda root, ds: pairs)

    if dataset == "3d":
        vol = np.zeros((2, 12, 13), dtype=np.float32)
        mask = np.zeros((2, 12, 13), dtype=np.uint16)
        mask[:, 2:8, 3:9] = 1
        monkeypatch.setattr(mod, "extract_3d_volume", lambda path, channels: vol)
        monkeypatch.setattr(mod, "load_3d_mask", lambda path: mask)
    else:
        nchan = len(channels)
        image = np.zeros((12, 13, nchan), dtype=np.float32) if nchan > 1 else np.zeros((12, 13), dtype=np.float32)
        mask = np.zeros((12, 13), dtype=np.uint16)
        mask[2:8, 3:9] = 1
        monkeypatch.setattr(mod, "extract_2d_image", lambda path, ds, channels: image)
        monkeypatch.setattr(mod, "load_2d_mask", lambda path: mask)

    cfg = mod.StarDistFineTuneConfig(project_root=tmp_path, epochs=2, steps_per_epoch=3, batch_size=1)
    assert cfg.channels == channels
    out = getattr(mod, function_name)(cfg)

    assert out == tmp_path / "models" / f"stardist_{dataset}" / "model"
    assert calls["config"][0]["train_batch_size"] == 1
    assert calls["train"][0][2]["epochs"] == 2
    assert calls["train"][0][2]["steps_per_epoch"] == 3
    config_json = tmp_path / "models" / f"stardist_{dataset}" / "training_config.json"
    assert json.loads(config_json.read_text(encoding="utf-8"))["epochs"] == 2


@pytest.mark.parametrize("module_name,function_name,dataset,channels", RUN_CASES)
def test_stardist_run_scripts(monkeypatch, tmp_path, module_name, function_name, dataset, channels):
    mod = importlib.import_module(module_name)
    calls = _install_fake_stardist(monkeypatch)
    image_path = tmp_path / "inputs" / "sample_a" / "image.ome.zarr"
    image_path.mkdir(parents=True)
    monkeypatch.setattr(mod, "list_input_images", lambda root, ds: [image_path])

    saved = []
    def fake_save(labels, out_dir, axes):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "labels.ome.zarr"
        path.mkdir(exist_ok=True)
        saved.append((np.asarray(labels), out_dir, axes))
        return path
    monkeypatch.setattr(mod, "save_label_outputs", fake_save)

    if dataset == "3d":
        vol = np.zeros((3, 10, 11), dtype=np.float32)
        monkeypatch.setattr(mod, "extract_3d_volume", lambda path, channels: vol)
    else:
        nchan = len(channels)
        img = np.zeros((10, 11, nchan), dtype=np.float32) if nchan > 1 else np.zeros((10, 11), dtype=np.float32)
        monkeypatch.setattr(mod, "extract_2d_image", lambda path, ds, channels: img)

    model_dir = tmp_path / "models" / f"stardist_{dataset}" / "model"
    model_dir.mkdir(parents=True)
    cfg = mod.StarDistRunConfig(project_root=tmp_path, model_dir=model_dir)
    outputs = getattr(mod, function_name)(cfg)

    assert len(outputs) == 1
    assert len(saved) == 1
    assert saved[0][2] == ("zyx" if dataset == "3d" else "yx")
    assert saved[0][0].shape == ((3, 10, 11) if dataset == "3d" else (10, 11))
    run_cfg = tmp_path / "results" / "segmentation_stardist" / dataset / "model" / "run_config.json"
    assert run_cfg.exists()


def test_stardist_load_model_uses_requested_directory(monkeypatch, tmp_path):
    _install_fake_stardist(monkeypatch)
    mod = importlib.import_module("PFT.core_prog_parts.stardist_run_2d_time_core")
    model_dir = tmp_path / "custom" / "my_model"
    model, returned = mod._load_model(tmp_path, model_dir)
    assert returned == model_dir
    assert model.name == "my_model"
    assert Path(model.basedir) == model_dir.parent
