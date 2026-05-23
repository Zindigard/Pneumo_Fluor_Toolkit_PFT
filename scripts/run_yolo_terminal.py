from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.yolo_run_2d_time_core import YOLORunConfig as Config2DTime, run_yolo_2d_time
from PFT.core_prog_parts.yolo_run_2d_wga_dapi_core import YOLORunConfig as Config2DWgaDapi, run_yolo_2d_wga_dapi
from PFT.core_prog_parts.yolo_run_3d_core import YOLORunConfig as Config3D, run_yolo_3d


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
