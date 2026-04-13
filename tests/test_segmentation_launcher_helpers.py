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


def test_default_nchan():
    mod = _load_launcher()
    assert mod.default_nchan("2d_time") == 1
    assert mod.default_nchan("2d_wga_dapi") == 2
    assert mod.default_nchan("3d_wga_dapi") == 2


def test_center_index():
    mod = _load_launcher()
    assert mod._center_index(1) == 0
    assert mod._center_index(4) == 2
    assert mod._center_index(5) == 2


def test_extract_model_name_from_output():
    mod = _load_launcher()
    p = Path("/tmp/some_folder/my_model")
    assert mod._extract_model_name_from_output(p) == "my_model"