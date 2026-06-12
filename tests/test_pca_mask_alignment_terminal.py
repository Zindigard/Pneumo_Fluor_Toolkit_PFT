from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest

from PFT.core_prog_parts.pca_mask_alignment_core import PCAMaskAlignmentConfig


def _load_launcher():
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_pca_mask_alignment_terminal.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_pca_mask_alignment_terminal",
        launcher_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_inputs(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    iterator = iter(values)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(iterator))


def test_launcher_imports_and_exposes_main() -> None:
    launcher = _load_launcher()
    assert hasattr(launcher, "main")
    assert hasattr(launcher, "build_config_interactive")
    assert hasattr(launcher, "choose_method")
    assert hasattr(launcher, "choose_dataset")


def test_terminal_choice_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()

    _set_inputs(monkeypatch, ["bad", "2"])
    assert launcher.ask_choice("Method", ["A", "B", "C"], default=0) == "B"

    _set_inputs(monkeypatch, [""])
    assert launcher.ask_yes_no("Continue", default=True) is True

    _set_inputs(monkeypatch, ["no"])
    assert launcher.ask_yes_no("Continue", default=True) is False

    _set_inputs(monkeypatch, ["invalid", "7"])
    assert launcher.ask_int("Minimum", default=5, minimum=1) == 7

    _set_inputs(monkeypatch, ["0.2", "1.5"])
    assert launcher.ask_float("Ratio", default=1.05, minimum=1.0) == 1.5

    _set_inputs(monkeypatch, ["nested/folder", "pca_aligned"])
    assert launcher.ask_folder_name("Folder", "default") == "pca_aligned"


def test_method_is_requested_before_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    calls: list[str] = []

    monkeypatch.setattr(launcher, "detect_project_root", lambda: Path("/tmp/project"))
    monkeypatch.setattr(launcher, "choose_method", lambda: calls.append("method") or "omnipose")
    monkeypatch.setattr(launcher, "choose_dataset", lambda: calls.append("dataset") or "2d_wga_dapi")
    monkeypatch.setattr(
        launcher,
        "choose_prediction_run",
        lambda *_args: Path("/tmp/project/results/run"),
    )
    monkeypatch.setattr(launcher, "choose_samples", lambda *_args: None)
    monkeypatch.setattr(launcher, "ask_folder_name", lambda *_args, **_kwargs: "pca_aligned")
    monkeypatch.setattr(launcher, "ask_int", lambda text, default, minimum=None: default)
    monkeypatch.setattr(launcher, "ask_float", lambda text, default, minimum=None: default)

    def fake_choice(title: str, options: list[str], default: int = 0) -> str:
        if "smaller" in title:
            return "Keep unchanged"
        if "overlap" in title:
            return "Keep larger cells first and fill only empty pixels"
        return options[default]

    yes_no_answers = iter([True, True, False, False])
    monkeypatch.setattr(launcher, "ask_choice", fake_choice)
    monkeypatch.setattr(launcher, "ask_yes_no", lambda *_args, **_kwargs: next(yes_no_answers))

    cfg = launcher.build_config_interactive()

    assert calls == ["method", "dataset"]
    assert cfg.method == "omnipose"
    assert cfg.dataset == "2d_wga_dapi"
    assert cfg.save_combined_mask is True
    assert cfg.save_individual_cells is True
    assert cfg.save_omezarr is False
    assert cfg.overwrite is False


@pytest.mark.parametrize(
    ("dataset", "function_name"),
    [
        ("2d_time", "run_pca_alignment_2d_time"),
        ("2d_wga_dapi", "run_pca_alignment_2d_wga_dapi"),
        ("3d", "run_pca_alignment_3d"),
    ],
)
def test_main_routes_to_dataset_specific_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    function_name: str,
) -> None:
    launcher = _load_launcher()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = PCAMaskAlignmentConfig(
        project_root=tmp_path,
        method="cellpose",
        dataset=dataset,
        prediction_run_dir=run_dir,
        save_omezarr=False,
    )

    called: list[str] = []
    monkeypatch.setattr(launcher, "build_config_interactive", lambda: cfg)
    monkeypatch.setattr(launcher, "print_config", lambda _cfg: None)
    monkeypatch.setattr(launcher, "ask_yes_no", lambda *_args, **_kwargs: True)

    for name in (
        "run_pca_alignment_2d_time",
        "run_pca_alignment_2d_wga_dapi",
        "run_pca_alignment_3d",
    ):
        monkeypatch.setattr(
            launcher,
            name,
            lambda _cfg, current=name: called.append(current) or [tmp_path / "out"],
        )

    assert launcher.main() == 0
    assert called == [function_name]
