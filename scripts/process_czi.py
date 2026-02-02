from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from PFT.core_prog_parts.io import list_czi_files, load_czi, load_czi_metadata_only
from PFT.core_prog_parts.save import export_2d, export_3d_metadata_only


@dataclass(frozen=True)
class Paths:
    data_2d_time: Path = Path(r"E:\2D_data_time")
    data_2d_wga_dapi: Path = Path(r"E:\2D_data_WGA_DAPI_DNA")
    data_3d: Path = Path(r"E:\20220225_HADA_NADA_TADA_40min")


PATHS = Paths()


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " (Y/n): " if default else " (y/N): "
    s = input(prompt + suffix).strip().lower()
    if not s:
        return default
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
        print(f"  [{i}] {opt}{d}")
    s = input("Enter number: ").strip()
    if s == "":
        return default
    idx = int(s)
    if idx < 0 or idx >= len(options):
        raise ValueError("Invalid choice")
    return idx


def _choose_folder(dataset: str) -> Path:
    """Let use default folder or enter a custom folder path."""
    if dataset == "2d_time":
        default_folder = PATHS.data_2d_time
    elif dataset == "2d_wga_dapi":
        default_folder = PATHS.data_2d_wga_dapi
    elif dataset == "3d":
        default_folder = PATHS.data_3d
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    use_default = _prompt_yes_no(f"Use default folder: {default_folder} ?", default=True)
    if use_default:
        return default_folder

    custom = input("Enter full folder path: ").strip().strip('"')
    folder = Path(custom)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return folder


def _choose_file_interactive(files: list[Path]) -> Path:
    print("\nAvailable files:")
    for i, f in enumerate(files):
        print(f"[{i:03d}] {f.name}")
    idx = _prompt_int("Choose file index", default=0)
    if idx < 0 or idx >= len(files):
        raise ValueError(f"Index {idx} out of range (0..{len(files)-1})")
    return files[idx]


def _run_one_2d(
    dataset: str,
    f: Path,
    visualize: bool,
    scalebar_um: float,
) -> Path:
    """
    Process a single 2D image:
    - loads array+meta
    - exports OME-Zarr + preview PNG + metadata files
    """
    arr, meta = load_czi(f)
    print("\n=== META ===")
    print(meta)

    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"
    wga_ch, dapi_ch = 1, 0

    out_dir = export_2d(
        arr=arr,
        meta=meta,
        dataset_name=dataset,
        preview_mode=preview_mode,
        visualize=visualize,
        save_preview_png=True,
        scalebar_um=scalebar_um,
        wga_ch=wga_ch,
        dapi_ch=dapi_ch,
    )
    return out_dir


def _run_all_2d(
    dataset: str,
    folder: Path,
    visualize: bool,
    scalebar_um: float,
    start: int = 0,
    limit: int | None = None,
    stop_on_error: bool = False,
) -> None:
    files = list_czi_files(folder)

    if start < 0 or start >= len(files):
        raise ValueError(f"--start {start} out of range (0..{len(files)-1})")

    files = files[start:]
    if limit is not None:
        files = files[:limit]

    print(f"\nBatch processing {len(files)} files from: {folder}")
    print(f"Dataset: {dataset}")

    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"
    wga_ch, dapi_ch = 1, 0  # fixed mapping

    ok = 0
    failed = 0
    for i, f in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            arr, meta = load_czi(f)
            export_2d(
                arr=arr,
                meta=meta,
                dataset_name=dataset,
                preview_mode=preview_mode,
                visualize=False,          # don’t pop up windows in batch
                save_preview_png=True,
                scalebar_um=scalebar_um,
                wga_ch=wga_ch,
                dapi_ch=dapi_ch,
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"ERROR on {f.name}: {e}")
            if stop_on_error:
                raise

    print(f"\nDone. Success: {ok} | Failed: {failed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi", "3d"], required=False)
    ap.add_argument("--folder", type=str, default=None, help="Override folder path.")
    ap.add_argument("--all", action="store_true", help="Process all files (2D only).")
    ap.add_argument("--index", type=int, default=None, help="Single file index.")
    ap.add_argument("--start", type=int, default=0, help="Start index for batch.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of files in batch.")
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--scalebar_um", type=float, default=5.0)
    args = ap.parse_args()

    # interactive  if no dataset 
    if args.dataset is None:
        dataset = ["2d_time", "2d_wga_dapi", "3d"][_prompt_choice("\nSelect dataset:", ["2d_time", "2d_wga_dapi", "3d"], 0)]
        folder = _choose_folder(dataset)
        scalebar_um = _prompt_float("Scalebar length in µm", default=5.0)

        if dataset in ("2d_time", "2d_wga_dapi"):
            mode = ["Process ONE file", "Process ALL files"][_prompt_choice("\nMode:", ["Process ONE file", "Process ALL files"], 0)]
            if mode == "Process ONE file":
                files = list_czi_files(folder)
                f = _choose_file_interactive(files)
                visualize = _prompt_yes_no("Visualize preview window?", default=False)
                out_dir = _run_one_2d(dataset, f, visualize=visualize, scalebar_um=scalebar_um)
                print("\nSaved to:", out_dir)
            else:
                start = _prompt_int("Start index", default=0)
                limit_s = input("Limit number of files (empty=all): ").strip()
                limit = int(limit_s) if limit_s else None
                stop_on_error = _prompt_yes_no("Stop on first error?", default=False)
                _run_all_2d(
                    dataset=dataset,
                    folder=folder,
                    visualize=False,
                    scalebar_um=scalebar_um,
                    start=start,
                    limit=limit,
                    stop_on_error=stop_on_error,
                )
        else:
            # 3D metadata only
            files = list_czi_files(folder)
            f = _choose_file_interactive(files)
            meta = load_czi_metadata_only(f)
            print("\n=== META ===")
            print(meta)
            out_dir = export_3d_metadata_only(meta, dataset_name="3d")
            print("\nSaved metadata to:", out_dir)

        return

  
    dataset = args.dataset
    folder = Path(args.folder) if args.folder else (
        PATHS.data_2d_time if dataset == "2d_time" else
        PATHS.data_2d_wga_dapi if dataset == "2d_wga_dapi" else
        PATHS.data_3d
    )

    if dataset in ("2d_time", "2d_wga_dapi"):
        if args.all:
            _run_all_2d(
                dataset=dataset,
                folder=folder,
                visualize=False,
                scalebar_um=args.scalebar_um,
                start=args.start,
                limit=args.limit,
                stop_on_error=args.stop_on_error,
            )
        else:
            files = list_czi_files(folder)
            f = files[args.index] if args.index is not None else files[0]
            out_dir = _run_one_2d(dataset, f, visualize=args.visualize, scalebar_um=args.scalebar_um)
            print("\nSaved to:", out_dir)
    else:
        files = list_czi_files(folder)
        f = files[args.index] if args.index is not None else files[0]
        meta = load_czi_metadata_only(f)
        print(meta)
        out_dir = export_3d_metadata_only(meta, dataset_name="3d")
        print("\nSaved metadata to:", out_dir)


if __name__ == "__main__":
    main()
