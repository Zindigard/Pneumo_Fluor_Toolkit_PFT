from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    The lookup is based on this script's physical location and therefore does
    not depend on the current working directory. An explicit error is raised
    when the expected repository layout cannot be found.
    """
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



import sys
from pathlib import Path

PROJECT_ROOT = _pft_project_root(Path(__file__).resolve())
SRC = PROJECT_ROOT / "src"

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.yolo_run_2d_time_core import YOLORunConfig as Config2DTime, run_yolo_2d_time
from PFT.core_prog_parts.segmentation.yolo_run_2d_wga_dapi_core import YOLORunConfig as Config2DWgaDapi, run_yolo_2d_wga_dapi
from PFT.core_prog_parts.segmentation.yolo_run_3d_core import YOLORunConfig as Config3D, run_yolo_3d


def main() -> int:
    root = find_project_root()
    print("\nChoose YOLO prediction dataset:")
    print("1) 2d_time")
    print("2) 2d_wga_dapi")
    print("3) 3d")
    choice = input("Choice [1-3]: ").strip()
    custom = input("Optional model path/folder, or press Enter for default: ").strip()
    model_ref = Path(custom) if custom else None
    if choice == "1":
        cfg = Config2DTime(project_root=root)
        if hasattr(cfg, "model_path"):
            cfg.model_path = model_ref
        elif hasattr(cfg, "model_dir"):
            cfg.model_dir = model_ref
        run_yolo_2d_time(cfg)
    elif choice == "2":
        cfg = Config2DWgaDapi(project_root=root)
        if hasattr(cfg, "model_path"):
            cfg.model_path = model_ref
        elif hasattr(cfg, "model_dir"):
            cfg.model_dir = model_ref
        run_yolo_2d_wga_dapi(cfg)
    elif choice == "3":
        cfg = Config3D(project_root=root)
        if hasattr(cfg, "model_path"):
            cfg.model_path = model_ref
        elif hasattr(cfg, "model_dir"):
            cfg.model_dir = model_ref
        run_yolo_3d(cfg)
    else:
        print("Invalid choice.")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
