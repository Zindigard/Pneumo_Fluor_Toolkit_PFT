from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TRAIN_CASES = [
    ("PFT.core_prog_parts.yolo_finetune_2d_time_core", "finetune_yolo_2d_time", "2d_time", (0,)),
    ("PFT.core_prog_parts.yolo_finetune_2d_wga_dapi_core", "finetune_yolo_2d_wga_dapi", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.yolo_finetune_3d_core", "finetune_yolo_3d", "3d", (0,)),
]
RUN_CASES = [
    ("PFT.core_prog_parts.yolo_run_2d_time_core", "run_yolo_2d_time", "2d_time", (0,)),
    ("PFT.core_prog_parts.yolo_run_2d_wga_dapi_core", "run_yolo_2d_wga_dapi", "2d_wga_dapi", (0, 1)),
    ("PFT.core_prog_parts.yolo_run_3d_core", "run_yolo_3d", "3d", (0,)),
]


def test_mask_to_yolo_segments_and_save_item(tmp_path):
    cv2 = pytest.importorskip("cv2")
    mod = importlib.import_module("PFT.core_prog_parts.yolo_finetune_2d_time_core")
    mask = np.zeros((32, 40), dtype=np.int32)
    mask[5:15, 6:18] = 1
    mask[18:28, 22:35] = 2
    rows = mod._mask_to_yolo_segments(mask, min_area=5)
    assert len(rows) == 2
    assert all(row.startswith("0 ") for row in rows)
    for row in rows:
        values = [float(v) for v in row.split()[1:]]
        assert len(values) >= 6
        assert all(0.0 <= v <= 1.0 for v in values)

    image_path = tmp_path / "images" / "sample.png"
    label_path = tmp_path / "labels" / "sample.txt"
    ok = mod._save_yolo_item(np.ones((32, 40), dtype=np.float32), mask, image_path, label_path)
    assert ok is True
    assert image_path.exists()
    assert label_path.exists()


def _install_fake_ultralytics(monkeypatch, create_weights=False):
    calls = {"init": [], "train": [], "predict": []}

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr)
        def cpu(self):
            return self
        def numpy(self):
            return self.arr

    class FakeMasks:
        def __init__(self, arr):
            self.data = FakeTensor(arr)

    class FakeResult:
        def __init__(self, arr):
            self.masks = FakeMasks(arr)

    class FakeYOLO:
        def __init__(self, model):
            calls["init"].append(str(model))
            self.model = model
        def train(self, **kwargs):
            calls["train"].append(kwargs)
            if create_weights:
                out = Path(kwargs["project"]) / kwargs["name"] / "weights"
                out.mkdir(parents=True, exist_ok=True)
                (out / "best.pt").write_bytes(b"weights")
            return {"ok": True}
        def predict(self, image, conf, iou, verbose=False):
            arr = np.asarray(image)
            calls["predict"].append((arr, conf, iou, verbose))
            h, w = arr.shape[:2]
            mask = np.zeros((1, h, w), dtype=np.uint8)
            mask[:, 2:-2, 3:-3] = 1
            return [FakeResult(mask)]

    pkg = types.ModuleType("ultralytics")
    pkg.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", pkg)
    return calls


@pytest.mark.parametrize("module_name,function_name,dataset,channels", TRAIN_CASES)
def test_yolo_finetune_scripts(monkeypatch, tmp_path, module_name, function_name, dataset, channels):
    mod = importlib.import_module(module_name)
    calls = _install_fake_ultralytics(monkeypatch, create_weights=True)
    data_yaml = tmp_path / "prepared" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("path: .\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_prepare_dataset", lambda cfg: data_yaml)

    cfg = mod.YOLOFineTuneConfig(project_root=tmp_path, epochs=2, batch=1, workers=0, device="cpu")
    assert cfg.channels == channels
    best = getattr(mod, function_name)(cfg)
    assert best.exists()
    assert calls["train"][0]["epochs"] == 2
    assert calls["train"][0]["batch"] == 1
    config_path = tmp_path / "models" / f"yolo_{dataset}" / "training_config.json"
    assert json.loads(config_path.read_text(encoding="utf-8"))["epochs"] == 2


@pytest.mark.parametrize("module_name,function_name,dataset,channels", RUN_CASES)
def test_yolo_run_scripts(monkeypatch, tmp_path, module_name, function_name, dataset, channels):
    mod = importlib.import_module(module_name)
    calls = _install_fake_ultralytics(monkeypatch)
    model_path = tmp_path / "models" / f"yolo_{dataset}" / "train" / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")

    image_path = tmp_path / "inputs" / "sample_a" / "image.ome.zarr"
    image_path.mkdir(parents=True)
    monkeypatch.setattr(mod, "list_input_images", lambda root, ds: [image_path])
    saved = []

    def fake_save(labels, out_dir, axes):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "labels.ome.zarr"
        path.mkdir(exist_ok=True)
        saved.append((np.asarray(labels), axes))
        return path
    monkeypatch.setattr(mod, "save_label_outputs", fake_save)

    if dataset == "3d":
        vol = np.zeros((3, 16, 18), dtype=np.float32)
        monkeypatch.setattr(mod, "extract_3d_volume", lambda path, channels: vol)
    else:
        nchan = len(channels)
        image = np.zeros((16, 18, nchan), dtype=np.float32) if nchan > 1 else np.zeros((16, 18), dtype=np.float32)
        monkeypatch.setattr(mod, "extract_2d_image", lambda path, ds, channels: image)

    cfg = mod.YOLORunConfig(project_root=tmp_path, model_path=model_path)
    assert cfg.channels == channels
    outputs = getattr(mod, function_name)(cfg)

    assert len(outputs) == 1
    assert saved[0][1] == ("zyx" if dataset == "3d" else "yx")
    assert saved[0][0].shape == ((3, 16, 18) if dataset == "3d" else (16, 18))
    assert len(calls["predict"]) == (3 if dataset == "3d" else 1)


def test_yolo_default_model_prefers_best_then_last(tmp_path):
    mod = importlib.import_module("PFT.core_prog_parts.yolo_run_2d_time_core")
    root = tmp_path / "models" / "yolo_2d_time" / "train" / "weights"
    root.mkdir(parents=True)
    last = root / "last.pt"
    last.write_bytes(b"last")
    assert mod._default_model(tmp_path) == last
    best = root / "best.pt"
    best.write_bytes(b"best")
    assert mod._default_model(tmp_path) == best
