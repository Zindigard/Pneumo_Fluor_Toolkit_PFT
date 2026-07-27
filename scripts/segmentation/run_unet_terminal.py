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



from pathlib import Path
import sys

_THIS = Path(__file__).resolve()

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.unet_run_core import (
    UNetRunConfig,
    default_input_root,
    default_model_path,
    default_out_root,
    find_omezarr_dirs,
    run_dataset,
)

"Use this script to run U-Net predictions. The output will be saved as OME-Zarr."

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
    print("\nChoose U-Net model/dataset to run:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    print("  3) 3d_25d   (2.5D U-Net on 3D images)")
    s = input("Choose number [1]: ").strip()
    if s == "2":
        return "2d_wga_dapi"
    if s == "3":
        return "3d_25d"
    return "2d_time"


def choose_targets(input_root: Path) -> tuple[bool, list[Path] | None]:
    zarrs = find_omezarr_dirs(input_root)
    if not zarrs:
        raise RuntimeError(f"No OME-Zarr folders found under: {input_root}")
    print("\nAvailable OME-Zarr inputs:")
    for i, p in enumerate(zarrs, 1):
        print(f"  {i}) {p}")
    print("  A) all")
    s = input("Choose one number or A for all [A]: ").strip().lower()
    if s in {"", "a", "all"}:
        return True, None
    idx = int(s) - 1
    if idx < 0 or idx >= len(zarrs):
        raise IndexError("Invalid selection")
    return False, [zarrs[idx]]


def ask_channels() -> tuple[int, ...] | None:
    s = input("3D image channels used during training, comma-separated, or Enter for all: ").strip()
    if not s:
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def main() -> int:
    root = find_project_root(Path(__file__).resolve())
    dataset = choose_dataset()

    cfg = UNetRunConfig(project_root=root, dataset=dataset)
    cfg.model_path = ask_path("Model path", default_model_path(root, dataset))
    cfg.input_root = ask_path("Input image root", default_input_root(root, dataset))
    cfg.out_root = ask_path("Output root", default_out_root(root, dataset))
    cfg.level = ask_int("OME-Zarr pyramid level", cfg.level)
    cfg.patch = ask_int("Patch size used by model", cfg.patch)
    cfg.threshold = ask_float("Prediction threshold", cfg.threshold)

    if dataset == "3d_25d":
        cfg.z_radius = ask_int("2.5D z-radius used during training", cfg.z_radius)
        cfg.channels = ask_channels()

    process_all, selected = choose_targets(cfg.input_root)

    print("\n=== RUN CONFIG ===")
    print(f"Project root: {root}")
    print(f"Dataset:      {dataset}")
    print(f"Model:        {cfg.model_path}")
    print(f"Input root:   {cfg.input_root}")
    print(f"Output root:  {cfg.out_root}")
    print(f"Level:        {cfg.level}")
    print(f"Patch:        {cfg.patch}")
    print(f"Threshold:    {cfg.threshold}")

    outputs = run_dataset(cfg, process_all=process_all, selected_zarrs=selected)

    print("\nPrediction finished. Saved OME-Zarr masks:")
    for p in outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
