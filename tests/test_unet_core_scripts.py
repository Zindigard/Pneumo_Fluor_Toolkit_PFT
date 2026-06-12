from __future__ import annotations

import importlib
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff


def _import_unet_modules(monkeypatch):
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(Model=object)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    for name in [
        "PFT.core_prog_parts.unet_train_2d_time_core",
        "PFT.core_prog_parts.unet_train_2d_wga_dapi_core",
        "PFT.core_prog_parts.unet_train_3d_25d_core",
        "PFT.core_prog_parts.unet_run_core",
    ]:
        sys.modules.pop(name, None)
    train2d = importlib.import_module("PFT.core_prog_parts.unet_train_2d_time_core")
    trainwga = importlib.import_module("PFT.core_prog_parts.unet_train_2d_wga_dapi_core")
    train3d = importlib.import_module("PFT.core_prog_parts.unet_train_3d_25d_core")
    run = importlib.import_module("PFT.core_prog_parts.unet_run_core")
    return train2d, trainwga, train3d, run


def test_unet_2d_normalization_and_image_extraction(monkeypatch, tmp_path):
    train2d, _, _, _ = _import_unet_modules(monkeypatch)
    x = np.arange(80, dtype=np.float32).reshape(8, 10)
    norm = train2d.normalize_image01(x, "percentile")
    assert norm.shape == x.shape
    assert norm.dtype == np.float32
    assert np.all((0 <= norm) & (norm <= 1))

    arr = np.stack([x, 2 * x], axis=0)
    monkeypatch.setattr(train2d, "load_ome_zarr", lambda path, level, as_numpy: (arr, "cyx"))
    one = train2d.ome_zarr_to_hwc_2d(tmp_path / "x", dataset="2d_time")
    dual = train2d.ome_zarr_to_hwc_2d(tmp_path / "x", dataset="2d_wga_dapi")
    assert one.shape == (8, 10, 1)
    assert dual.shape == (8, 10, 2)


def test_unet_2d_masks_pairs_split_and_patch(monkeypatch, tmp_path):
    train2d, _, _, _ = _import_unet_modules(monkeypatch)
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    for name in ("a", "b"):
        (image_root / name / "image.ome.zarr").mkdir(parents=True)
        (mask_root / name).mkdir(parents=True)
        mask = np.zeros((32, 40), dtype=np.uint8)
        mask[8:24, 10:30] = 1
        tiff.imwrite(mask_root / name / "mask.tif", mask)

    cfg = train2d.UNet2DTrainConfig(project_root=tmp_path, image_root=image_root, mask_root=mask_root, patch=16, val_split=0.5)
    pairs = train2d.list_2d_training_pairs(cfg)
    assert len(pairs) == 2
    tr, val = train2d.split_pairs(pairs, cfg)
    assert len(tr) == 1 and len(val) == 1

    img = np.zeros((32, 40, 1), dtype=np.float32)
    mask = train2d.read_binary_mask_2d(mask_root / "a" / "mask.tif")
    patch_img, patch_mask = train2d.sample_patch_2d(img, mask, cfg, random.Random(0))
    assert patch_img.shape == (16, 16, 1)
    assert patch_mask.shape == (16, 16)


def test_unet_wga_wrapper_maps_configuration(monkeypatch, tmp_path):
    train2d, trainwga, _, _ = _import_unet_modules(monkeypatch)
    captured = {}
    monkeypatch.setattr(trainwga, "train_2d_binary_unet", lambda cfg: captured.setdefault("cfg", cfg) or {"ok": Path("x")})
    cfg = trainwga.UNet2DWgaDapiTrainConfig(project_root=tmp_path, epochs=3)
    result = trainwga.train_2d_wga_dapi_unet(cfg)
    mapped = captured["cfg"]
    assert mapped.dataset == "2d_wga_dapi"
    assert mapped.epochs == 3


def test_unet_25d_input_and_mask_loading(monkeypatch, tmp_path):
    _, _, train3d, _ = _import_unet_modules(monkeypatch)
    cfg = train3d.UNet25DTrainConfig(project_root=tmp_path, z_radius=1, channels=(0, 1), patch=8)
    img = np.zeros((2, 4, 12, 14), dtype=np.float32)
    for c in range(2):
        for z in range(4):
            img[c, z] = c * 10 + z
    plane = train3d.make_25d_input_slice(img, 0, cfg)
    assert plane.shape == (12, 14, 6)
    assert np.all(plane[..., 0] == 0)
    assert np.all(plane[..., 2] == 0)
    assert np.all(plane[..., 4] == 1)

    mask_path = tmp_path / "mask.tif"
    tiff.imwrite(mask_path, np.ones((4, 12, 14), dtype=np.uint8))
    mask = train3d.load_3d_mask_zyx(mask_path, expected_zyx=(4, 12, 14))
    assert mask.shape == (4, 12, 14)
    assert mask.dtype == np.uint8


def test_unet_run_padding_tiled_prediction_and_paths(monkeypatch, tmp_path):
    _, _, _, run = _import_unet_modules(monkeypatch)
    img = np.arange(10 * 13, dtype=np.float32).reshape(10, 13, 1)
    padded, pads = run.pad_to_patch(img, 8)
    assert padded.shape == (16, 16, 1)
    assert pads == (6, 3)

    class FakeModel:
        def predict(self, batch, verbose=0):
            return np.ones(batch.shape[:3] + (1,), dtype=np.float32)

    cfg = run.UNetRunConfig(project_root=tmp_path, patch=8, threshold=0.5)
    pred = run.predict_2d_tiled(FakeModel(), img, cfg)
    assert pred.shape == (10, 13)
    assert np.all(pred == 1)
    assert run.default_model_path(tmp_path, "2d_time").name == "u_net_2d_time_best.keras"
    assert run.default_model_path(tmp_path, "3d_25d").name == "u_net_3d_25d_best.keras"


def test_unet_run_dataset_routing(monkeypatch, tmp_path):
    _, _, _, run = _import_unet_modules(monkeypatch)
    model_path = tmp_path / "model.keras"
    model_path.write_text("x", encoding="utf-8")
    target = tmp_path / "input" / "sample" / "image.ome.zarr"
    target.mkdir(parents=True)
    monkeypatch.setattr(run, "load_unet_model", lambda path: object())
    monkeypatch.setattr(run, "find_omezarr_dirs", lambda root: [target])
    calls = []
    monkeypatch.setattr(run, "run_2d_time_unet_on_omezarr", lambda path, cfg, model: calls.append(("time", path)) or tmp_path / "time.zarr")
    monkeypatch.setattr(run, "run_2d_wga_dapi_unet_on_omezarr", lambda path, cfg, model: calls.append(("wga", path)) or tmp_path / "wga.zarr")
    monkeypatch.setattr(run, "run_3d_25d_unet_on_omezarr", lambda path, cfg, model: calls.append(("3d", path)) or tmp_path / "3d.zarr")

    for dataset, expected in [("2d_time", "time"), ("2d_dpa_wagi", "wga"), ("3d", "3d")]:
        cfg = run.UNetRunConfig(project_root=tmp_path, dataset=dataset, model_path=model_path, input_root=tmp_path / "input")
        outputs = run.run_dataset(cfg)
        assert outputs
        assert calls[-1][0] == expected
