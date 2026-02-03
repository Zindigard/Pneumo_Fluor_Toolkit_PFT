from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from PFT.core_prog_parts.io import list_czi_files, load_czi, load_czi_metadata_only
from PFT.core_prog_parts.save import export_2d, export_3d_metadata_only
from PFT.core_prog_parts.Ome_Zarr import save_ome_zarr_next_to_outputs

REPO_ROOT = Path(__file__).resolve().parents[3] 
TRAIN_COLLECTION_DIR = REPO_ROOT / "results" / "training_files"

@dataclass(frozen=True)
class Paths:
    data_2d_time: Path = Path(r"E:\2D_data_time")
    data_2d_wga_dapi: Path = Path(r"E:\2D_data_WGA_DAPI_DNA")
    data_3d: Path = Path(r"E:\20220225_HADA_NADA_TADA_40min")

PATHS = Paths()

def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " (Y/n): " if default else " (y/N): "
    s = input(prompt + suffix).strip().lower()
    if not s: return default
    return s in ("y", "yes", "1", "true")

def _prompt_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} (empty={default}): ").strip()
    return float(s) if s else default

def _prompt_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} (empty={default}): ").strip()
    return int(s) if s else default

def _prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    print(prompt)
    for i, opt in enumerate(options):
        d = " (default)" if i == default else ""
        print(f"   [{i}] {opt}{d}")
    s = input("Enter number: ").strip()
    if s == "": return default
    idx = int(s)
    if idx < 0 or idx >= len(options): raise ValueError("Invalid choice")
    return idx

def _choose_folder(dataset: str) -> Path:
    if dataset == "2d_time": default_folder = PATHS.data_2d_time
    elif dataset == "2d_wga_dapi": default_folder = PATHS.data_2d_wga_dapi
    elif dataset == "3d": default_folder = PATHS.data_3d
    else: raise ValueError(f"Unknown dataset: {dataset}")

    if _prompt_yes_no(f"Use default folder: {default_folder} ?", default=True):
        return default_folder
    custom = input("Enter full folder path: ").strip().strip('"')
    folder = Path(custom)
    if not folder.exists(): raise FileNotFoundError(f"Folder not found: {folder}")
    return folder

def _choose_file_interactive(files: list[Path]) -> Path:
    print("\nAvailable files:")
    for i, f in enumerate(files):
        print(f"[{i:03d}] {f.name}")
    idx = _prompt_int("Choose file index", default=0)
    if idx < 0 or idx >= len(files):
        raise ValueError(f"Index {idx} out of range (0..{len(files)-1})")
    return files[idx]


def _run_one_2d(dataset: str, f: Path, visualize: bool, scalebar_um: float) -> Path:
    """
    Processes one file and saves a copy to the central training repository.
    """
    arr, meta = load_czi(f)
    print("\n=== META ===")
    print(meta)

    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"
    
    out_dir = export_2d(
        arr=arr, meta=meta, dataset_name=dataset,
        preview_mode=preview_mode, visualize=visualize,
        save_preview_png=True, scalebar_um=scalebar_um,
        wga_ch=1, dapi_ch=0
    )

    # collect for training
    TRAIN_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    file_target_dir = TRAIN_COLLECTION_DIR / f.stem
    save_ome_zarr_next_to_outputs(file_target_dir, arr, meta, overwrite=True)
    
    return out_dir

def _run_all_2d(dataset: str, folder: Path, scalebar_um: float, 
                start: int = 0, limit: int | None = None, stop_on_error: bool = False) -> None:
    """
    Batch processes files and adds each to the training collection.
    """
    files = list_czi_files(folder)
    files = files[start:]
    if limit is not None: files = files[:limit]

    print(f"\nBatch processing {len(files)} files. Dataset: {dataset}")
    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"

    ok, failed = 0, 0
    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            arr, meta = load_czi(f)
            export_2d(
                arr=arr, meta=meta, dataset_name=dataset,
                preview_mode=preview_mode, visualize=False,
                save_preview_png=True, scalebar_um=scalebar_um,
                wga_ch=1, dapi_ch=0
            )
            # Training 
            TRAIN_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
            save_ome_zarr_next_to_outputs(TRAIN_COLLECTION_DIR / f.stem, arr, meta, overwrite=True)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"ERROR on {f.name}: {e}")
            if stop_on_error: raise
    print(f"\nDone. Success: {ok} | Failed: {failed}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi", "3d"])
    ap.add_argument("--folder", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--scalebar_um", type=float, default=5.0)
    args = ap.parse_args()

    if args.dataset is None:
        dataset = ["2d_time", "2d_wga_dapi", "3d"][_prompt_choice("\nSelect dataset:", ["2d_time", "2d_wga_dapi", "3d"], 0)]
        folder = _choose_folder(dataset)
        scalebar_um = _prompt_float("Scalebar length in µm", default=5.0)

        if dataset in ("2d_time", "2d_wga_dapi"):
            mode = ["Process ONE file", "Process ALL files"][_prompt_choice("\nMode:", ["Process ONE file", "Process ALL files"], 0)]
            if mode == "Process ONE file":
                f = _choose_file_interactive(list_czi_files(folder))
                out_dir = _run_one_2d(dataset, f, visualize=_prompt_yes_no("Visualize?", False), scalebar_um=scalebar_um)
                print("\nSaved to:", out_dir)
            else:
                _run_all_2d(dataset, folder, scalebar_um, _prompt_int("Start", 0), 
                            None, _prompt_yes_no("Stop on error?", False))
        else:
            f = _choose_file_interactive(list_czi_files(folder))
            meta = load_czi_metadata_only(f)
            print(meta)
            export_3d_metadata_only(meta, dataset_name="3d")
        return

    dataset = args.dataset
    folder = Path(args.folder) if args.folder else (PATHS.data_2d_time if dataset == "2d_time" else PATHS.data_2d_wga_dapi if dataset == "2d_wga_dapi" else PATHS.data_3d)

    if dataset in ("2d_time", "2d_wga_dapi"):
        if args.all:
            _run_all_2d(dataset, folder, args.scalebar_um, args.start, args.limit, args.stop_on_error)
        else:
            files = list_czi_files(folder)
            f = files[args.index] if args.index is not None else files[0]
            _run_one_2d(dataset, f, args.visualize, args.scalebar_um)
    else:
        f = list_czi_files(folder)[args.index or 0]
        meta = load_czi_metadata_only(f)
        print(meta)
        export_3d_metadata_only(meta, dataset_name="3d")

if __name__ == "__main__":
    main()