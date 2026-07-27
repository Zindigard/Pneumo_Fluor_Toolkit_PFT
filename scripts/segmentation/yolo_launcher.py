from __future__ import annotations
# Permit direct execution from a source checkout after scripts were grouped into subpackages.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()
for _pft_parent in [_PFT_SCRIPT_FILE.parent, *_PFT_SCRIPT_FILE.parents]:
    _pft_src = _pft_parent / "src"
    if (_pft_src / "PFT").exists():
        if str(_pft_src) not in _pft_sys.path:
            _pft_sys.path.insert(0, str(_pft_src))
        break

def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Locate the repository root independently of script nesting depth."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "PFT").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return _PFTPath.cwd()


import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT_GUESS = _pft_project_root(_THIS_FILE)
_SRC = _PROJECT_ROOT_GUESS / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.yolo_train import build_train_config_interactive, train_yolo_model
from PFT.core_prog_parts.segmentation.yolo_run import build_run_config_interactive, run_yolo_dataset


""" YOLO training and prediction workflows."""


def prompt_choice(title: str, options: list[str], default: int = 0) -> str:
    """Ask the user to choose one option from a numbered list."""
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        tag = " (default)" if i - 1 == default else ""
        print(f"  {i}) {opt}{tag}")
    while True:
        s = input(f"Choose number [1-{len(options)}] or Enter for default: ").strip()
        if s == "":
            return options[default]
        try:
            idx = int(s) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(options):
            return options[idx]
        print("Invalid choice.")


def main() -> None:
    """Run the interactive YOLO launcher."""
    project_root = find_project_root(_THIS_FILE)
    print(f"Project root: {project_root}")

    mode = prompt_choice("Choose workflow:", ["train / fine-tune", "run / predict"], default=0)

    if mode.startswith("train"):
        cfg = build_train_config_interactive(project_root)
        out_dir = train_yolo_model(cfg)
        print(f"\nTraining outputs saved to: {out_dir}")
    else:
        cfg = build_run_config_interactive(project_root)
        out_dir = run_yolo_dataset(cfg)
        print(f"\nPrediction outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
