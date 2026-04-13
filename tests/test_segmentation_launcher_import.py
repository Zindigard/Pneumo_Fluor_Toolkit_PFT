from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_launcher():
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "segmentation_launcher.py"
    spec = importlib.util.spec_from_file_location("segmentation_launcher", launcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launcher_loads():
    mod = _load_launcher()
    assert hasattr(mod, "main")
    assert hasattr(mod, "compute_binary_metrics")
    assert hasattr(mod, "save_labels_as_omezarr")