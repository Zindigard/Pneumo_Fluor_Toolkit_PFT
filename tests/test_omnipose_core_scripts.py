from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff

TRAIN_CASES = [
    ("PFT.core_prog_parts.omnipose_finetune_2d_time_core", "finetune_omnipose_2d_time", "2d_time", (0,)),
    ("PFT.core_prog_parts.omnipose_finetune_2d_wga_dapi_core", "finetune_omnipose_2d_wga_dapi", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.omnipose_finetune_3d_core", "finetune_omnipose_3d", "3d", (0,)),
]
RUN_CASES = [
    ("PFT.core_prog_parts.omnipose_run_2d_time_core", "run_omnipose_2d_time", "2d_time", (0,)),
    ("PFT.core_prog_parts.omnipose_run_2d_wga_dapi_core", "run_omnipose_2d_wga_dapi", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.omnipose_run_3d_core", "run_omnipose_3d", "3d", (0,)),
]


@pytest.mark.parametrize("module_name,function_name,dataset,channels", TRAIN_CASES)
def test_omnipose_prepare_and_train_command(monkeypatch, tmp_path, module_name, function_name, dataset, channels):
    mod = importlib.import_module(module_name)
    pairs = [(tmp_path / "image.ome.zarr", tmp_path / "mask.tif", "sample_a")]
    monkeypatch.setattr(mod, "collect_training_pairs", lambda root, ds: pairs)

    if dataset == "3d":
        vol = np.zeros((2, 8, 9), dtype=np.float32)
        mask = np.zeros((2, 8, 9), dtype=np.int32)
        mask[:, 2:6, 3:7] = 1
        monkeypatch.setattr(mod, "extract_3d_volume", lambda path, channels: vol)
        monkeypatch.setattr(mod, "load_3d_mask", lambda path: mask)
    else:
        nchan = len(channels)
        img = np.zeros((8, 9, nchan), dtype=np.float32) if nchan > 1 else np.zeros((8, 9), dtype=np.float32)
        mask = np.zeros((8, 9), dtype=np.int32)
        mask[2:6, 3:7] = 1
        monkeypatch.setattr(mod, "extract_2d_image", lambda path, ds, channels: img)
        monkeypatch.setattr(mod, "load_2d_mask", lambda path: mask)

    commands = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, check, cwd: commands.append((cmd, check, cwd)))
    cfg = mod.OmniposeFineTuneConfig(project_root=tmp_path, n_epochs=2, batch_size=1, use_gpu=False)
    assert cfg.channels == channels
    model_path = getattr(mod, function_name)(cfg)

    prepared = tmp_path / "models" / f"omnipose_{dataset}" / "prepared_training_data"
    if dataset == "3d":
        assert len(list(prepared.glob("*_img.tif"))) == 2
        assert len(list(prepared.glob("*_masks.tif"))) == 2
    else:
        assert (prepared / "sample_a_img.tif").exists()
        assert (prepared / "sample_a_masks.tif").exists()
    assert commands and "--train" in commands[0][0]
    assert "--use_gpu" not in commands[0][0]
    assert "--n_epochs" in commands[0][0]
    config_path = tmp_path / "models" / f"omnipose_{dataset}" / "training_config.json"
    assert json.loads(config_path.read_text(encoding="utf-8"))["n_epochs"] == 2
    assert model_path.exists()


class _FakeOmniModel:
    def __init__(self):
        self.calls = []

    def eval(self, img, **kwargs):
        self.calls.append((np.asarray(img), kwargs))
        labels = np.zeros(np.asarray(img).shape[:2], dtype=np.uint16)
        labels[1:-1, 1:-1] = 1
        return labels, {}, None


@pytest.mark.parametrize("module_name,function_name,dataset,channels", RUN_CASES)
def test_omnipose_run_scripts(monkeypatch, tmp_path, module_name, function_name, dataset, channels):
    mod = importlib.import_module(module_name)
    model = _FakeOmniModel()
    model_ref = tmp_path / "models" / f"omnipose_{dataset}" / "model.pt"
    model_ref.parent.mkdir(parents=True)
    model_ref.write_bytes(b"x")
    monkeypatch.setattr(mod, "_build_model", lambda cfg: (model, model_ref))

    image_path = tmp_path / "inputs" / "sample_a" / "image.ome.zarr"
    image_path.mkdir(parents=True)
    monkeypatch.setattr(mod, "list_input_images", lambda root, ds: [image_path])
    saved = []

    def fake_save(labels, out_dir, axes):
        out_dir.mkdir(parents=True, exist_ok=True)
        zarr = out_dir / "labels.ome.zarr"
        zarr.mkdir(exist_ok=True)
        saved.append((np.asarray(labels), axes))
        return zarr
    monkeypatch.setattr(mod, "save_label_outputs", fake_save)

    if dataset == "3d":
        vol = np.zeros((3, 10, 11), dtype=np.float32)
        monkeypatch.setattr(mod, "extract_3d_volume", lambda path, channels: vol)
    else:
        nchan = len(channels)
        img = np.zeros((10, 11, nchan), dtype=np.float32) if nchan > 1 else np.zeros((10, 11), dtype=np.float32)
        monkeypatch.setattr(mod, "extract_2d_image", lambda path, ds, channels: img)

    cfg = mod.OmniposeRunConfig(project_root=tmp_path, gpu=False)
    assert cfg.channels == channels
    outputs = getattr(mod, function_name)(cfg)
    assert len(outputs) == 1
    assert saved[0][1] == ("zyx" if dataset == "3d" else "yx")
    assert saved[0][0].shape == ((3, 10, 11) if dataset == "3d" else (10, 11))
    assert len(model.calls) == (3 if dataset == "3d" else 1)


def test_omnipose_eval_extracts_first_tuple_element():
    mod = importlib.import_module("PFT.core_prog_parts.omnipose_run_2d_time_core")
    model = _FakeOmniModel()
    cfg = mod.OmniposeRunConfig(project_root=Path("."), gpu=False)
    labels = mod._eval_2d(model, np.zeros((7, 8), dtype=np.float32), cfg)
    assert labels.shape == (7, 8)
    assert labels.dtype == np.uint16
