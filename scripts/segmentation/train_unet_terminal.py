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
import sys

_THIS = Path(__file__).resolve()
for p in [_THIS.parent, *_THIS.parents]:
    if (p / "src" / "PFT").exists():
        src = p / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        break

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import UNet2DTrainConfig, train_2d_time_unet
from PFT.core_prog_parts.segmentation.unet_train_2d_wga_dapi_core import UNet2DWgaDapiTrainConfig, train_2d_wga_dapi_unet
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import UNet25DTrainConfig, train_3d_25d_unet

"Use this script to run U-Net fine-tuning. The output will be saved as OME-Zarr."


def ask_int(text: str, default: int) -> int:
    s = input(f"{text} [{default}]: ").strip()
    return default if s == "" else int(s)


def ask_float(text: str, default: float) -> float:
    s = input(f"{text} [{default}]: ").strip()
    return default if s == "" else float(s)


def ask_path(text: str, default: Path) -> Path:
    s = input(f"{text}\nDefault: {default}\nPath or Enter: ").strip().strip('"')
    return default if s == "" else Path(s)


def choose_dataset() -> str:
    print("\nChoose U-Net training dataset:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    print("  3) 3d_25d   (2.5D U-Net, nearby z-slices)")
    s = input("Choose number [1]: ").strip()
    if s == "2":
        return "2d_wga_dapi"
    if s == "3":
        return "3d_25d"
    return "2d_time"


def ask_channels() -> tuple[int, ...] | None:
    s = input("3D image channels to use, comma-separated, or Enter for all channels: ").strip()
    if not s:
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def main() -> int:
    root = find_project_root(Path(__file__).resolve())
    dataset = choose_dataset()

    if dataset == "2d_time":
        cfg = UNet2DTrainConfig(project_root=root, dataset="2d_time")
        cfg.image_root = ask_path("Training image root", root / "results" / "img" / "filtered" / "2d_time")
        cfg.mask_root = ask_path("Training mask root", root / "results" / "training_files" / "U-net" / "2d_time")
        cfg.model_root = root / "models" / "u_net_2d_time"
    elif dataset == "2d_wga_dapi":
        cfg = UNet2DWgaDapiTrainConfig(project_root=root, dataset="2d_wga_dapi")
        cfg.image_root = ask_path("Training image root", root / "results" / "img" / "filtered" / "2d_wga_dapi")
        cfg.mask_root = ask_path("Training mask root", root / "results" / "training_files" / "U-net" / "2d_wga_dapi")
        cfg.model_root = root / "models" / "u_net_2d_wga_dapi"
    else:
        cfg = UNet25DTrainConfig(project_root=root)
        cfg.image_root = ask_path("3D training image root", root / "results" / "img" / "3d_data")
        cfg.mask_root = ask_path("3D training mask root", root / "results" / "3d_label")
        cfg.model_root = root / "models" / "u_net_3d_25d"
        cfg.z_radius = ask_int("2.5D z-radius: 1 means z-1, z, z+1", cfg.z_radius)
        cfg.channels = ask_channels()

    print("\nTraining parameters:")
    cfg.level = ask_int("OME-Zarr pyramid level", cfg.level)
    cfg.patch = ask_int("Patch size", cfg.patch)
    cfg.batch = ask_int("Batch size", cfg.batch)
    cfg.epochs = ask_int("Epochs", cfg.epochs)
    cfg.steps_per_epoch = ask_int("Steps per epoch", cfg.steps_per_epoch)
    cfg.val_steps = ask_int("Validation steps", cfg.val_steps)
    cfg.lr = ask_float("Learning rate", cfg.lr)
    cfg.base_filters = ask_int("Base filters", cfg.base_filters)

    print("\n=== TRAINING CONFIG ===")
    print(f"Project root: {root}")
    print(f"Dataset:      {dataset}")
    print(f"Image root:   {cfg.image_root}")
    print(f"Mask root:    {cfg.mask_root}")
    print(f"Model root:   {cfg.model_root}")
    print(f"Patch/batch:  {cfg.patch}/{cfg.batch}")
    print(f"Epochs:       {cfg.epochs}")

    if dataset == "2d_time":
        outputs = train_2d_time_unet(cfg)
    elif dataset == "2d_wga_dapi":
        outputs = train_2d_wga_dapi_unet(cfg)
    else:
        outputs = train_3d_25d_unet(cfg)

    print("\nTraining finished. Saved files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
