from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_launcher():
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "segmentation_launcher.py"
    spec = importlib.util.spec_from_file_location("segmentation_launcher", launcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_binary_metrics_perfect_overlap():
    mod = _load_launcher()

    pred = np.array([[0, 1], [0, 1]], dtype=np.int32)
    gt = np.array([[0, 2], [0, 2]], dtype=np.int32)

    metrics = mod.compute_binary_metrics(pred, gt)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["dice_fg"] == 1.0
    assert metrics["iou_fg"] == 1.0
    assert metrics["precision_fg"] == 1.0
    assert metrics["recall_fg"] == 1.0


def test_compute_binary_metrics_shape_mismatch():
    mod = _load_launcher()

    pred = np.zeros((4, 4), dtype=np.int32)
    gt = np.zeros((3, 4), dtype=np.int32)

    try:
        mod.compute_binary_metrics(pred, gt)
    except ValueError as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")