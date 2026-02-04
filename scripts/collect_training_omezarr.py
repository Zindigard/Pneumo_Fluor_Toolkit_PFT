from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_image_omezarr_parents(dataset_root: Path) -> list[Path]:
    """
    Returns a list of sample folders that contain an 'image.ome.zarr' directory.
    Example returned path:
      .../results/img/2d_wga_dapi/WT_THY_noCSP_ROI3_SIM
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
