from __future__ import annotations

from pathlib import Path

from PFT.core_prog_parts import common_paths as cp


def test_normalize_dataset_aliases():
    assert cp.normalize_dataset_name(" 2D_TIME ") == "2d_time"
    assert cp.normalize_dataset_name("2d_dpa_wagi") == "2d_wga_dapi"
    assert cp.normalize_dataset_name("3D") == "3d"


def test_ensure_dir_and_result_paths(tmp_path):
    target = cp.ensure_dir(tmp_path / "a" / "b")
    assert target.is_dir()
    assert cp.results_img_dir(tmp_path) == tmp_path / "results" / "img"
    assert cp.filtered_img_root(tmp_path) == tmp_path / "results" / "img" / "filtered"
    assert cp.results_filters_dir(tmp_path) == tmp_path / "results" / "Filters"
    assert cp.dataset_img_dir("2d_dpa_wga", tmp_path) == tmp_path / "results" / "img" / "2d_wga_dapi"


def test_find_project_root_from_nested_file(tmp_path):
    root = tmp_path / "repo"
    nested = root / "src" / "PFT" / "core_prog_parts" / "x.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert cp.find_project_root(nested) == root
