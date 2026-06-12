from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff

try:
    from PFT.core_prog_parts import pca_mask_alignment_core as pca
except ImportError:
    from PFT.core_prog_parts import pca_maskt_core as pca


def _diagonal_instance(shape=(80, 100), label=1):
    mask = np.zeros(shape, dtype=np.int32)
    for x in range(20, 75):
        y = int(18 + 0.55 * (x - 20))
        mask[max(0, y - 3): y + 4, x] = label
    return mask


def test_pca_method_dataset_aliases_and_validation(tmp_path):
    assert pca.normalize_method("cell-pose") == "cellpose"
    assert pca.normalize_method("omni") == "omnipose"
    assert pca.normalize_dataset("wagi_dapi") == "2d_wga_dapi"
    assert pca.normalize_dataset("3d_sim") == "3d"
    with pytest.raises(ValueError):
        pca.normalize_method("unknown")

    run = tmp_path / "run"
    run.mkdir()
    cfg = pca.PCAMaskAlignmentConfig(
        project_root=tmp_path,
        method="star_dist",
        dataset="time",
        prediction_run_dir=run,
    )
    cfg.validate()
    assert cfg.method == "stardist"
    assert cfg.dataset == "2d_time"


def test_pca_orientation_and_rotation_are_horizontal():
    mask = _diagonal_instance() > 0
    before = pca._pca_orientation(mask)
    assert before["pca_valid"] is True
    assert abs(float(before["angle_deg"])) > 10
    rotated = pca._rotate_to_horizontal(mask, float(before["angle_deg"]), padding=4)
    after = pca._pca_orientation(rotated)
    assert abs(float(after["angle_deg"])) < 3.0


def test_binary_mask_is_converted_to_instances():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[3:10, 4:12] = 1
    mask[25:35, 30:45] = 1
    labels, changed = pca._ensure_instance_labels(mask)
    assert changed is True
    assert set(np.unique(labels)) == {0, 1, 2}


def test_prediction_run_discovery_for_all_methods(tmp_path):
    for method in pca.SUPPORTED_METHODS:
        run = tmp_path / "results" / f"segmentation_{method}" / "2d_time" / "model_a"
        sample = run / "sample_a"
        sample.mkdir(parents=True)
        tiff.imwrite(sample / "labels.tif", np.zeros((8, 9), dtype=np.uint16))
        discovered = pca.discover_prediction_runs(tmp_path, method, "2d_time")
        assert run.resolve() in discovered


def test_align_mask_file_creates_combined_and_cell_outputs(tmp_path):
    run = tmp_path / "run"
    sample = run / "sample_a"
    sample.mkdir(parents=True)
    mask = _diagonal_instance()
    mask[55:70, 70:90] = 2
    mask_path = sample / "labels.tif"
    tiff.imwrite(mask_path, mask.astype(np.uint16))

    cfg = pca.PCAMaskAlignmentConfig(
        project_root=tmp_path,
        method="cellpose",
        dataset="2d_time",
        prediction_run_dir=run,
        save_omezarr=False,
        overwrite=True,
    )
    out = pca.align_mask_file(mask_path, sample, cfg)
    assert (out / "labels_pca_aligned.tif").exists()
    assert (out / "pca_alignment.csv").exists()
    assert len(list((out / "cells").glob("*.tif"))) == 2
    summary = json.loads((out / "alignment_summary.json").read_text(encoding="utf-8"))
    assert summary["n_objects"] == 2
    assert summary["output_axes"] == "yx"


def test_run_pca_alignment_filters_samples(tmp_path):
    run = tmp_path / "run"
    for name in ("a", "b"):
        sample = run / name
        sample.mkdir(parents=True)
        tiff.imwrite(sample / "labels.tif", _diagonal_instance().astype(np.uint16))
    cfg = pca.PCAMaskAlignmentConfig(
        project_root=tmp_path,
        method="stardist",
        dataset="2d_time",
        prediction_run_dir=run,
        sample_names=("b",),
        save_omezarr=False,
    )
    outputs = pca.run_pca_alignment_2d_time(cfg)
    assert outputs == [run.resolve() / "b" / "pca_aligned"]
    assert not (run / "a" / "pca_aligned").exists()
