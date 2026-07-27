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

_THIS_FILE = Path(__file__).resolve()
for _p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (_p / "src" / "PFT").exists():
        _SRC_DIR = _p / "src"
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        break

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root
import argparse
import shutil
from pathlib import Path

"""Script to collect OME-Zarr directories from the datasets, rename them, and split into training/validation folders."""

def find_image_omezarr_parents(dataset_root: Path) -> list[Path]:
    """
    Returns a list of sample folders that contain an 'image.ome.zarr' directory.
    Example returned path:
      
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    hits: list[Path] = []
    for zarr_dir in dataset_root.rglob("image.ome.zarr"):
        if zarr_dir.is_dir():
            hits.append(zarr_dir.parent)
    return sorted(set(hits))


def move_image_zarr(sample_dir: Path, dst_sample_dir: Path, overwrite: bool) -> None:
    """
    Move sample
    """
    src_zarr = sample_dir / "image.ome.zarr"
    if not src_zarr.is_dir():
        raise FileNotFoundError(f"Missing source zarr folder: {src_zarr}")

    dst_sample_dir.mkdir(parents=True, exist_ok=True)
    dst_zarr = dst_sample_dir / "image.ome.zarr"

    if dst_zarr.exists():
        if not overwrite:
            return
        shutil.rmtree(dst_zarr)

    shutil.move(str(src_zarr), str(dst_zarr))



def main() -> None:
    ap = argparse.ArgumentParser(
        )
    ap.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi"], required=False)
    ap.add_argument(
        "--repo_root",
        type=str,
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT",
        help="Repo/project root that contains results/...",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination if it already exists.")
    args = ap.parse_args()

    dataset = args.dataset
    if dataset is None:
        print("Select dataset:")
        print("  [0] 2d_time")
        print("  [1] 2d_wga_dapi")
        s = input("Enter number (default=0): ").strip()
        dataset = "2d_wga_dapi" if s == "1" else "2d_time"

    repo_root = Path(args.repo_root)
    src_dataset_root = repo_root / "results" / "img" / dataset
    dst_dataset_root = repo_root / "results" / "training_files" / dataset

    sample_dirs = find_image_omezarr_parents(src_dataset_root)
    if not sample_dirs:
        print(f"No image.ome.zarr found under: {src_dataset_root}")
        return

    total = len(sample_dirs)

    print(f"Found {total} sample(s) under {src_dataset_root}")
    print(f"Destination base: {dst_dataset_root}")

    ok, failed, skipped = 0, 0, 0

    for i, sample_dir in enumerate(sample_dirs, start=1):
        sample_name = sample_dir.name  
        dst_sample_dir = dst_dataset_root / sample_name
        dst_zarr = dst_sample_dir / "image.ome.zarr"

        prefix = f"[{i}/{total}] {sample_name}"

        try:
            if dst_zarr.exists() and not args.overwrite:
                skipped += 1
                print(f"{prefix}  SKIP (exists)")
                continue

            move_image_zarr(
                sample_dir,
                dst_sample_dir,
                overwrite=args.overwrite,
            )
            ok += 1
            print(f"{prefix}  OK")

        except Exception as e:
            failed += 1
            print(f"{prefix}  ERROR: {e}")

    print(f"Done. OK: {ok} | Skipped: {skipped} | Failed: {failed} | Total: {total}")



if __name__ == "__main__":
    main()
