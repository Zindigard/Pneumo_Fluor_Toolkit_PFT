from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.omnipose_finetune_2d_time_core import OmniposeFineTuneConfig as Config2DTime, finetune_omnipose_2d_time
from PFT.core_prog_parts.omnipose_finetune_2d_wga_dapi_core import OmniposeFineTuneConfig as Config2DWgaDapi, finetune_omnipose_2d_wga_dapi
from PFT.core_prog_parts.omnipose_finetune_3d_core import OmniposeFineTuneConfig as Config3D, finetune_omnipose_3d


def ask_int(text: str, default: int) -> int:
    value = input(f"{text} [{default}]: ").strip()
    return default if not value else int(value)


def ask_float(text: str, default: float) -> float:
    value = input(f"{text} [{default}]: ").strip()
    return default if not value else float(value)


def main() -> int:
    root = find_project_root()
    print("\nChoose Omnipose fine-tuning dataset:")
    print("1) 2d_time")
    print("2) 2d_wga_dapi")
    print("3) 3d")
    choice = input("Choice [1-3]: ").strip()
    if choice == "1":
        cfg = Config2DTime(project_root=root)
        cfg.epochs = ask_int("Epochs", getattr(cfg, "epochs", getattr(cfg, "n_epochs", 100))) if hasattr(cfg, "epochs") else cfg.n_epochs
        if hasattr(cfg, "n_epochs"):
            cfg.n_epochs = ask_int("Epochs", cfg.n_epochs)
        finetune_omnipose_2d_time(cfg)
    elif choice == "2":
        cfg = Config2DWgaDapi(project_root=root)
        if hasattr(cfg, "epochs"):
            cfg.epochs = ask_int("Epochs", cfg.epochs)
        if hasattr(cfg, "n_epochs"):
            cfg.n_epochs = ask_int("Epochs", cfg.n_epochs)
        finetune_omnipose_2d_wga_dapi(cfg)
    elif choice == "3":
        cfg = Config3D(project_root=root)
        if hasattr(cfg, "epochs"):
            cfg.epochs = ask_int("Epochs", cfg.epochs)
        if hasattr(cfg, "n_epochs"):
            cfg.n_epochs = ask_int("Epochs", cfg.n_epochs)
        finetune_omnipose_3d(cfg)
    else:
        print("Invalid choice.")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
