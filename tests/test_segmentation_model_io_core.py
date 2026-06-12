from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff

from PFT.core_prog_parts import segmentation_model_io_core as smio


def test_basic_helpers_and_channel_selection():
    assert smio.center_index(1) == 0
    assert smio.center_index(6) == 3
    assert smio.safe_sample_name(Path("/tmp/sample/image.ome.zarr")) == "sample"
    assert smio.safe_sample_name(Path("/tmp/sample/pred_mask.ome.zarr")) == "sample"
    assert smio.dataset_channels("2d_wga_dapi") == (0, 1)
    assert smio.dataset_channels("2d_time") == (0,)

    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    norm = smio.normalize_float32(x, 0, 100)
    assert norm.dtype == np.float32
    assert np.isclose(norm.min(), 0)
    assert np.isclose(norm.max(), 1)
    assert smio.to_uint8(x).dtype == np.uint8


def test_time_and_axis_reordering():
    arr = np.zeros((2, 3, 4, 5), dtype=np.float32)
    out, axes = smio.take_first_time(arr, "tcyx")
    assert out.shape == (3, 4, 5)
    assert axes == "cyx"

    yxc = np.zeros((4, 5, 2), dtype=np.float32)
    cyx, axes = smio.move_yx_last(yxc, "yxc")
    assert cyx.shape == (2, 4, 5)
    assert axes == "cyx"

    with pytest.raises(ValueError):
        smio.move_yx_last(np.zeros((2, 3)), "cz")


def test_extract_2d_image_from_mocked_ome(monkeypatch, tmp_path):
    arr = np.zeros((2, 3, 8, 10), dtype=np.float32)  # CZYX
    arr[0, 1] = np.arange(80).reshape(8, 10)
    arr[1, 1] = 2 * np.arange(80).reshape(8, 10)
    monkeypatch.setattr(smio, "load_ome", lambda path, level=0: (arr, "czyx"))

    single = smio.extract_2d_image(tmp_path / "x", "2d_time")
    assert single.shape == (8, 10)
    assert single.dtype == np.float32

    dual = smio.extract_2d_image(tmp_path / "x", "2d_wga_dapi")
    assert dual.shape == (8, 10, 2)


def test_extract_3d_volume_from_mocked_ome(monkeypatch, tmp_path):
    arr = np.arange(2 * 3 * 8 * 10, dtype=np.float32).reshape(2, 3, 8, 10)
    monkeypatch.setattr(smio, "load_ome", lambda path, level=0: (arr, "czyx"))
    vol = smio.extract_3d_volume(tmp_path / "x", channels=(0, 1))
    assert vol.shape == (3, 8, 10, 2)
    assert vol.dtype == np.float32


def test_mask_loading_and_candidate_order(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    m2 = np.zeros((2, 6, 7), dtype=np.uint16)
    m2[1, 2:4, 2:5] = 3
    tiff.imwrite(sample / "labels.tif", m2)
    loaded2 = smio.load_2d_mask(sample / "labels.tif")
    assert loaded2.shape == (6, 7)
    assert loaded2.dtype == np.int32

    m3 = np.zeros((3, 6, 7), dtype=np.uint16)
    tiff.imwrite(sample / "mask.tif", m3)
    loaded3 = smio.load_3d_mask(sample / "mask.tif")
    assert loaded3.shape == (3, 6, 7)

    candidates = list(smio.mask_candidates(sample))
    assert sample / "mask.tif" in candidates
    assert sample / "labels.tif" in candidates


def test_resolve_list_and_collect_training_pairs(tmp_path):
    sample = "sample_a"
    image = tmp_path / "results" / "denoised_images" / "2d_time" / sample / "image.ome.zarr"
    image.mkdir(parents=True)
    mask_dir = tmp_path / "results" / "segmentation_finetuning_masks" / "2d_time" / sample
    mask_dir.mkdir(parents=True)
    tiff.imwrite(mask_dir / "mask.tif", np.ones((5, 6), dtype=np.uint8))

    assert smio.resolve_image(tmp_path, "2d_time", sample) == image
    assert smio.list_input_images(tmp_path, "2d_time") == [image]
    pairs = smio.collect_training_pairs(tmp_path, "2d_time")
    assert pairs == [(image, mask_dir / "mask.tif", sample)]


def test_model_and_prediction_roots_and_json(tmp_path):
    model = smio.model_root(tmp_path, "stardist", "2d_time")
    pred = smio.prediction_root(tmp_path, "stardist", "2d_time", "model")
    assert model.is_dir()
    assert pred.is_dir()

    @dataclass
    class Payload:
        root: Path
        values: np.ndarray

    out = tmp_path / "payload.json"
    smio.write_json(out, Payload(tmp_path, np.array([1, 2])))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["root"] == str(tmp_path)
    assert data["values"] == [1, 2]


def test_save_label_outputs_calls_writer_with_expected_contract(monkeypatch, tmp_path):
    captured = {}

    def fake_save(out_path, arr, axes, **kwargs):
        captured["out_path"] = Path(out_path)
        captured["arr"] = np.asarray(arr)
        captured["axes"] = axes
        Path(out_path).mkdir(parents=True, exist_ok=True)
        return Path(out_path)

    monkeypatch.setattr(smio, "save_ome_zarr", fake_save)
    labels = np.zeros((8, 9), dtype=np.int32)
    labels[2:5, 3:7] = 1
    out = smio.save_label_outputs(labels, tmp_path / "sample", axes="yx")
    assert (tmp_path / "sample" / "labels.tif").exists()
    assert out == tmp_path / "sample" / "labels.ome.zarr"
    assert captured["out_path"] == out
    assert captured["axes"] == "yx"
    assert np.array_equal(captured["arr"], labels.astype(np.uint16))
