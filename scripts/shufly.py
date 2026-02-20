from __future__ import annotations
import random
import shutil
from pathlib import Path

"""Script to collect OME-Zarr directories from the datasets, rename them, and split into training/validation folders."""

BASE = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\training_files")
DATASETS = [
    BASE / "2d_time",
    BASE / "2d_wga_dapi",
]

TRAIN_DIRNAME = "training_data"
VAL_DIRNAME = "validation_data"

TRAIN_FRAC = 0.60 
RANDOM_SEED = 42   


def is_omezarr_dir(p: Path) -> bool:
    return p.is_dir() and p.name.endswith(".ome.zarr")


def safe_move(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists:\n  {dst}\nRefusing to overwrite.")
    shutil.move(str(src), str(dst))


def safe_rename(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Rename target already exists:\n  {dst}\nRefusing to overwrite.")
    src.rename(dst)


def collect_and_rename_omezarr(dataset_root: Path) -> list[Path]:
    """
    For each sample folder inside dataset_root:
      - if it contains "image.ome.zarr", rename it to "<sample_folder_name>_omezarr"
      - if it already contains "<sample_folder_name>_omezarr", keep it
    Returns the list of *final* omezarr directories (paths).
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    omezarr_dirs: list[Path] = []

    for sample_dir in sorted(dataset_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        # Skip our split folders if script was run before
        if sample_dir.name in (TRAIN_DIRNAME, VAL_DIRNAME):
            continue

        standard = sample_dir / "image.ome.zarr"
        renamed = sample_dir / f"{sample_dir.name}_omezarr"

        if is_omezarr_dir(standard):
            safe_rename(standard, renamed)
            omezarr_dirs.append(renamed)
        elif is_omezarr_dir(renamed):
            omezarr_dirs.append(renamed)
        else:
            continue

    return omezarr_dirs


def split_into_train_val(dataset_root: Path, omezarr_dirs: list[Path]) -> None:
    train_dir = dataset_root / TRAIN_DIRNAME
    val_dir = dataset_root / VAL_DIRNAME

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    if not omezarr_dirs:
        print(f"[INFO] No OME-Zarr dirs found under: {dataset_root}")
        return

    rng = random.Random(RANDOM_SEED)
    items = list(omezarr_dirs)
    rng.shuffle(items)

    n_total = len(items)
    n_train = int(round(n_total * TRAIN_FRAC))
    n_train = max(1, min(n_train, n_total - 1)) if n_total >= 2 else n_total

    train_items = items[:n_train]
    val_items = items[n_train:]

    print(f"[INFO] Dataset: {dataset_root.name}")
    print(f"       Total: {n_total} | Train: {len(train_items)} | Val: {len(val_items)}")

    for src in train_items:
        dst = train_dir / src.name
        safe_move(src, dst)

    for src in val_items:
        dst = val_dir / src.name
        safe_move(src, dst)


def main() -> None:
    for dataset_root in DATASETS:
        print(f"\n=== Processing {dataset_root} ===")
        omezarr_dirs = collect_and_rename_omezarr(dataset_root)
        split_into_train_val(dataset_root, omezarr_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
