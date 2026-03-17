from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from PFT.core_prog_parts.io import list_czi_files, load_czi
from PFT.core_prog_parts.save import export_2d, export_3d
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs

"""
Process CZI files, extract metadata, save previews + OME-Zarr exports.
2D training collection OME-Zarr is explicitly single-scale (no pyramid)
3D exports can create multiscale pyramids 
"""

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_COLLECTION_DIR = REPO_ROOT / "results" / "training_files"


@dataclass(frozen=True)
class Paths:
    data_2d_time: Path = Path(r"E:\2D_data_time")
    data_2d_wga_dapi: Path = Path(r"E:\2D_data_WGA_DAPI_DNA")
    data_3d: Path = Path(r"E:\3d_data")


PATHS = Paths()


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " (Y/n): " if default else " (y/N): "
    s = input(prompt + suffix).strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true")


def prompt_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} (empty={default}): ").strip()
    return float(s) if s else default


def prompt_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} (empty={default}): ").strip()
    return int(s) if s else default


def prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    print(prompt)
    for i, opt in enumerate(options):
        d = " (default)" if i == default else ""
        print(f"   [{i}] {opt}{d}")
    s = input("Enter number: ").strip()
    if s == "":
        return default
    idx = int(s)
    if idx < 0 or idx >= len(options):
        raise ValueError("Invalid choice")
    return idx


def choose_folder(dataset: str) -> Path:
    if dataset == "2d_time":
        default_folder = PATHS.data_2d_time
    elif dataset == "2d_wga_dapi":
        default_folder = PATHS.data_2d_wga_dapi
    elif dataset == "3d":
        default_folder = PATHS.data_3d
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if prompt_yes_no(f"Use default folder: {default_folder} ?", default=True):
        return default_folder

    custom = input("Enter full folder path: ").strip().strip('"')
    folder = Path(custom)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return folder


def choose_file_interactive(files: list[Path]) -> Path:
    print("\nAvailable files:")
    for i, f in enumerate(files):
        print(f"[{i:03d}] {f.name}")
    idx = prompt_int("Choose file index", default=0)
    if idx < 0 or idx >= len(files):
        raise ValueError(f"Index {idx} out of range (0..{len(files)-1})")
    return files[idx]


def run_one_2d(dataset: str, f: Path, visualize: bool, scalebar_um: float) -> Path:
    """
    Processes one 2D file, writes previews + image.ome.zarr in results/img,
    """
    arr, meta = load_czi(f)
    print("\n=== META ===")
    print(meta)

    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"

    out_dir = export_2d(
        arr=arr,
        meta=meta,
        dataset_name=dataset,
        preview_mode=preview_mode,
        visualize=visualize,
        save_preview_png=True,
        scalebar_um=scalebar_um,
        wga_ch=1,
        dapi_ch=0,
    )

    TRAIN_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    file_target_dir = TRAIN_COLLECTION_DIR / f.stem
    save_ome_zarr_next_to_outputs(
        file_target_dir,
        arr,
        meta,
        overwrite=True,
        pyramid_3d=False,          #
        pyramid_max_layer=0,
        pyramid_downscale=2,
    )

    return out_dir


def run_all_2d(
    dataset: str,
    folder: Path,
    scalebar_um: float,
    start: int = 0,
    limit: int | None = None,
    stop_on_error: bool = False,
) -> None:
    """
    Batch processes 2D files and adds each to the training collection.
    """
    files = list_czi_files(folder)
    files = files[start:]
    if limit is not None:
        files = files[:limit]

    print(f"\nBatch processing {len(files)} files. Dataset: {dataset}")
    preview_mode = "wga_dapi" if dataset == "2d_wga_dapi" else "time_blue"

    ok, failed = 0, 0
    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {f.name}")
        try:
            arr, meta = load_czi(f)
            export_2d(
                arr=arr,
                meta=meta,
                dataset_name=dataset,
                preview_mode=preview_mode,
                visualize=False,
                save_preview_png=True,
                scalebar_um=scalebar_um,
                wga_ch=1,
                dapi_ch=0,
            )

            TRAIN_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
            save_ome_zarr_next_to_outputs(
                TRAIN_COLLECTION_DIR / f.stem,
                arr,
                meta,
                overwrite=True,
                pyramid_3d=False,
                pyramid_max_layer=0,
                pyramid_downscale=2,
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"ERROR on {f.name}: {e}")
            if stop_on_error:
                raise

    print(f"\nDone. Success: {ok} | Failed: {failed}")


def run_all_3d(
    base_folder: Path,
    *,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
) -> None:
    """
    Batch export 3D.
    Pyramid is controlled by ome_zarr.py and enabled automatically for 3D axes.
    """
    expected = [
        "20220218_dynamic",
        "20220225_HADA_NADA_TADA_40min",
        "20220318_HADA_NADA_TADA_40min",
        "20220330_HADA_NADA_TADA_0min",
    ]

    if not base_folder.exists():
        print(f"ERROR: base 3D folder not found: {base_folder}")
        return

    ok, failed = 0, 0
    out_base = REPO_ROOT / "results" / "img"

    for ds in expected:
        ds_folder = base_folder / ds
        if not ds_folder.exists():
            print(f"WARNING: missing dataset folder: {ds_folder}")
            continue

        files = sorted(ds_folder.glob("*.czi"))
        if not files:
            print(f"WARNING: no .czi files found in {ds_folder}")
            continue

        print(f"\n=== 3D dataset: {ds} | files: {len(files)} ===")
        for i, f in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] {f.name}")
            try:
                arr, meta = load_czi(f)

                export_3d(
                    arr=arr,
                    meta=meta,
                    dataset_folder=ds,
                    out_base=out_base,
                    save_omezarr=True,
                    overwrite_omezarr=True,
                )
                ok += 1
            except Exception as e:
                failed += 1
                print(f"ERROR on {f.name}: {e}")

    print(f"\n3D export done. Success: {ok} | Failed: {failed}")


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

    ap.add_argument("--pyramid_max_layer", type=int, default=2, help="3D pyramid depth (0 disables pyramid)")
    ap.add_argument("--pyramid_downscale", type=int, default=2, help="3D downscale factor per level (usually 2)")

    args = ap.parse_args()

    if args.dataset is None:
        dataset = ["2d_time", "2d_wga_dapi", "3d"][
            prompt_choice("\nSelect dataset:", ["2d_time", "2d_wga_dapi", "3d"], 0)
        ]
        folder = choose_folder(dataset)
        scalebar_um = prompt_float("Scalebar length in µm", default=5.0)

        if dataset in ("2d_time", "2d_wga_dapi"):
            mode = ["Process ONE file", "Process ALL files"][
                prompt_choice("\nMode:", ["Process ONE file", "Process ALL files"], 0)
            ]
            if mode == "Process ONE file":
                f = choose_file_interactive(list_czi_files(folder))
                out_dir = run_one_2d(
                    dataset,
                    f,
                    visualize=prompt_yes_no("Visualize?", False),
                    scalebar_um=scalebar_um,
                )
                print("\nSaved to:", out_dir)
            else:
                run_all_2d(
                    dataset,
                    folder,
                    scalebar_um,
                    prompt_int("Start", 0),
                    None,
                    prompt_yes_no("Stop on error?", False),
                )
        else:
            run_all_3d(folder, pyramid_max_layer=args.pyramid_max_layer, pyramid_downscale=args.pyramid_downscale)
        return

    dataset = args.dataset
    folder = (
        Path(args.folder)
        if args.folder
        else (
            PATHS.data_2d_time
            if dataset == "2d_time"
            else PATHS.data_2d_wga_dapi
            if dataset == "2d_wga_dapi"
            else PATHS.data_3d
        )
    )

    if dataset in ("2d_time", "2d_wga_dapi"):
        if args.all:
            run_all_2d(dataset, folder, args.scalebar_um, args.start, args.limit, args.stop_on_error)
        else:
            files = list_czi_files(folder)
            f = files[args.index] if args.index is not None else files[0]
            run_one_2d(dataset, f, args.visualize, args.scalebar_um)
    else:
        run_all_3d(folder, pyramid_max_layer=args.pyramid_max_layer, pyramid_downscale=args.pyramid_downscale)


if __name__ == "__main__":
    main()