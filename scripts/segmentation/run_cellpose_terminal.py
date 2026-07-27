
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


from pathlib import Path

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.cellpose_run_2d_time_core import CellposeRun2DTimeConfig, run_cellpose_dataset as run_2d_time
from PFT.core_prog_parts.segmentation.cellpose_run_2d_wga_dapi_core import CellposeRun2DWgaDapiConfig, run_cellpose_dataset as run_2d_wga_dapi
from PFT.core_prog_parts.segmentation.cellpose_run_3d_core import CellposeRun3DConfig, run_cellpose_dataset as run_3d


"Use this script to run Cellpose predictions from the terminal. The output will be saved as OME-Zarr."

def _yes_no(prompt: str, default: bool = True) -> bool:
    suffix = 'Y/n' if default else 'y/N'
    ans = input(f'{prompt} [{suffix}]: ').strip().lower()
    if not ans:
        return default
    return ans in {'y', 'yes', '1', 'true'}


def _float_or_none(prompt: str, default: float | None) -> float | None:
    shown = 'None' if default is None else str(default)
    ans = input(f'{prompt} [{shown}]: ').strip()
    if not ans:
        return default
    if ans.lower() in {'none', 'no', 'n'}:
        return None
    return float(ans)


def _int_value(prompt: str, default: int) -> int:
    ans = input(f'{prompt} [{default}]: ').strip()
    return int(ans) if ans else default


def _float_value(prompt: str, default: float) -> float:
    ans = input(f'{prompt} [{default}]: ').strip()
    return float(ans) if ans else default


def _model_path_or_none() -> str | None:
    ans = input('Model path [empty = newest model from models/cellpose_<dataset>]: ').strip().strip('"')
    return ans or None


def main() -> None:
    project_root = find_project_root()
    print('\nCellpose prediction launcher')
    print(f'Project root: {project_root}')
    print('\nChoose dataset to run:')
    print('  1) 2d_time')
    print('  2) 2d_wga_dapi')
    print('  3) 3d, slice-wise prediction saved as ZYX OME-Zarr')
    choice = input('Selection [1/2/3]: ').strip()

    model_path = _model_path_or_none()
    gpu = _yes_no('Use GPU?', True)
    diameter = _float_or_none('Cell diameter', None)
    flow_threshold = _float_value('Flow threshold', 0.4)
    cellprob_threshold = _float_value('Cell probability threshold', 0.0)
    min_size = _int_value('Minimum object size', 15)
    batch_size = _int_value('Batch size', 8)

    if choice == '1':
        cfg = CellposeRun2DTimeConfig(
            project_root=project_root, model_path=model_path, gpu=gpu,
            diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size, batch_size=batch_size,
        )
        out = run_2d_time(cfg)
    elif choice == '2':
        cfg = CellposeRun2DWgaDapiConfig(
            project_root=project_root, model_path=model_path, gpu=gpu,
            diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size, batch_size=batch_size,
        )
        out = run_2d_wga_dapi(cfg)
    elif choice == '3':
        cfg = CellposeRun3DConfig(
            project_root=project_root, model_path=model_path, gpu=gpu,
            diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size, batch_size=batch_size,
        )
        out = run_3d(cfg)
    else:
        raise SystemExit('Unknown choice.')

    print(f'\nSaved Cellpose OME-Zarr predictions to:\n{out}')


if __name__ == '__main__':
    main()
