from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import zarr

pytest.importorskip("ome_zarr")


def _load_launcher():
    launcher_path = Path(__file__).resolve().parents[1] / "scripts" / "segmentation_launcher.py"
    spec = importlib.util.spec_from_file_location("segmentation_launcher", launcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_level0(zarr_dir):
    root = zarr.open_group(str(zarr_dir), mode="r")
    assert "0" in root
    return np.asarray(root["0"])


def test_save_labels_as_omezarr_creates_output(tmp_path):
    mod = _load_launcher()

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 4:8] = 1

    out_dir = tmp_path / "segmented_sample"
    zarr_path = mod.save_labels_as_omezarr(out_dir, labels, source_path="dummy.tif")

    assert zarr_path.exists()
    arr_back = _load_level0(zarr_path)
    assert arr_back.shape == labels.shape
    assert np.array_equal(arr_back, labels)