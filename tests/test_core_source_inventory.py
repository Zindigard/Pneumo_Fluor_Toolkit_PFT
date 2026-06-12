from __future__ import annotations

import ast
from pathlib import Path


def _core_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "PFT" / "core_prog_parts"


def test_every_core_python_file_parses():
    core = _core_dir()
    files = sorted(core.glob("*.py"))
    assert files, f"No core files found in {core}"
    errors: list[str] = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
        except Exception as exc:  # pragma: no cover - assertion message path
            errors.append(f"{path.name}: {exc}")
    assert not errors, "Core syntax errors:\n" + "\n".join(errors)


def test_expected_core_families_are_present():
    names = {p.name for p in _core_dir().glob("*.py")}
    expected = {
        "common_paths.py",
        "decoder_omezar.py",
        "omezarr_utils.py",
        "segmentation_model_io_core.py",
        "cellpose_run_2d_time_core.py",
        "cellpose_run_2d_wga_dapi_core.py",
        "cellpose_run_3d_core.py",
        "cellpose_finetune_2d_time_core.py",
        "cellpose_finetune_2d_wga_dapi_core.py",
        "cellpose_finetune_3d_core.py",
        "stardist_run_2d_time_core.py",
        "stardist_run_2d_wga_dapi_core.py",
        "stardist_run_3d_core.py",
        "omnipose_run_2d_time_core.py",
        "omnipose_run_2d_wga_dapi_core.py",
        "omnipose_run_3d_core.py",
        "yolo_run_2d_time_core.py",
        "yolo_run_2d_wga_dapi_core.py",
        "yolo_run_3d_core.py",
        "unet_run_core.py",
    }
    assert expected <= names
    assert {"pca_mask_alignment_core.py", "pca_maskt_core.py"} & names
