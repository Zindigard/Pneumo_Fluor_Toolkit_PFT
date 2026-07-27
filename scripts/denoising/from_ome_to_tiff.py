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
import shutil
import numpy as np
import tifffile as tif

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr_volume_csyx


RESULTS_DIR = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results")

BASE_DIRS = {
    "2d_time": RESULTS_DIR
    / r"Filters\Local_high_threshold\2d_time\with_BM3D\p99.2_k3f40_s370_k10f40",

    "2d_wga_dapi": RESULTS_DIR
    / r"Filters\Local_high_threshold\2d_wga_dapi\with_BM3D\p99.2_k3f40_s370_k10f40",
}

ORIGINAL_RAW_RGB_DIRS = {
    "2d_time": RESULTS_DIR / r"img\2d_time",
    "2d_wga_dapi": RESULTS_DIR / r"img\2d_wga_dapi",
}

OUTPUT_ROOT = RESULTS_DIR / "denoised_images"

LEVEL = 0

NORMALIZE_LOW_PERCENTILE = 1.0
NORMALIZE_HIGH_PERCENTILE = 99.8


def normalize_to_original_dtype_range_per_image(
    arr_csyx: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> np.ndarray:
   
    original_dtype = arr_csyx.dtype

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        out_min = info.min
        out_max = info.max
        output_dtype = original_dtype
    else:
        out_min = 0.0
        out_max = 1.0
        output_dtype = np.float32

    arr = arr_csyx.astype(np.float32, copy=False)
    out = np.zeros(arr.shape, dtype=output_dtype)

    for c in range(arr.shape[0]):
        for s in range(arr.shape[1]):
            img = arr[c, s]

            lo = np.percentile(img, p_low)
            hi = np.percentile(img, p_high)

            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out[c, s] = 0
                continue

            norm = (img - lo) / (hi - lo)
            norm = np.clip(norm, 0.0, 1.0)

            scaled = norm * (out_max - out_min) + out_min
            out[c, s] = scaled.astype(output_dtype)

    return out



def make_rgb_csyx(
    arr_csyx: np.ndarray,
    dataset_group: str,
) -> np.ndarray:
   
    if arr_csyx.ndim != 4:
        raise ValueError(f"Expected shape (C,S,Y,X), got {arr_csyx.shape}")

    c, s, y, x = arr_csyx.shape
    rgb = np.zeros((s, y, x, 3), dtype=arr_csyx.dtype)

    if dataset_group == "2d_time":
        if c >= 1:
            rgb[..., 2] = arr_csyx[0]  # blue

    elif dataset_group == "2d_wga_dapi":
        if c >= 1:
            rgb[..., 1] = arr_csyx[0]  # green
        if c >= 2:
            rgb[..., 2] = arr_csyx[1]  # blue

    else:
        if c >= 1:
            rgb[..., 0] = arr_csyx[0]
            rgb[..., 1] = arr_csyx[0]
            rgb[..., 2] = arr_csyx[0]

    return rgb



def find_ome_zarr_dirs(parent_dir: Path) -> list[Path]:
    
    parent_dir = Path(parent_dir)
    zarr_dirs = []

    for folder in parent_dir.iterdir():
        if not folder.is_dir():
            continue

        direct = folder / "image.ome.zarr"

        if direct.exists() and direct.is_dir():
            zarr_dirs.append(direct)
            continue

        found = list(folder.rglob("*.ome.zarr"))
        zarr_dirs.extend([p for p in found if p.is_dir()])

    return sorted(set(zarr_dirs))



def find_original_raw_rgb_file(
    original_group_dir: Path,
    dataset_name: str,
) -> Path | None:
   
    dataset_dir = Path(original_group_dir) / dataset_name

    if not dataset_dir.exists():
        return None

    exact_candidates = [
        dataset_dir / "image_raw_rgb.tif",
        dataset_dir / "image_raw_rgb.tiff",
    ]

    for p in exact_candidates:
        if p.exists() and p.is_file():
            return p

    patterns = [
        "*image_raw_rgb*.tif",
        "*image_raw_rgb*.tiff",
        "*raw_rgb*.tif",
        "*raw_rgb*.tiff",
    ]

    found: list[Path] = []
    for pattern in patterns:
        found.extend(dataset_dir.rglob(pattern))

    found = sorted([p for p in found if p.is_file()])

    if not found:
        return None

    return found[0]


def copy_original_raw_rgb_if_exists(
    dataset_group: str,
    dataset_name: str,
    dataset_output_dir: Path,
) -> None:
   
    original_group_dir = ORIGINAL_RAW_RGB_DIRS.get(dataset_group)

    if original_group_dir is None:
        print("       original raw RGB: no source folder configured")
        return

    src = find_original_raw_rgb_file(
        original_group_dir=original_group_dir,
        dataset_name=dataset_name,
    )

    if src is None:
        print("       original raw RGB: NOT FOUND")
        return

    dst = dataset_output_dir / "original_raw_rgb.tif"
    shutil.copy2(src, dst)

    print(f"       original raw RGB copied from: {src}")
    print(f"       original raw RGB copied to:   {dst}")



def export_omezarr_folder_to_grouped_normalized_colored_tiffs(
    dataset_group: str,
    base_dir: Path,
    output_root: Path,
    *,
    level: int = 0,
) -> None:
 
    base_dir = Path(base_dir)
    output_root = Path(output_root)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base folder does not exist:\n{base_dir}")

    group_output_dir = output_root / dataset_group
    group_output_dir.mkdir(parents=True, exist_ok=True)

    zarr_dirs = find_ome_zarr_dirs(base_dir)

    if not zarr_dirs:
        print(f"[WARNING] No OME-Zarr folders found in:\n{base_dir}")
        return

    print("\n========================================")
    print(f"Dataset group: {dataset_group}")
    print(f"Base folder:   {base_dir}")
    print(f"Output folder: {group_output_dir}")
    print(f"Found OME-Zarr folders: {len(zarr_dirs)}")
    print("========================================\n")

    for zarr_dir in zarr_dirs:
        dataset_name = zarr_dir.parent.name

        dataset_output_dir = group_output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        denoised_norm_rgb_out = dataset_output_dir / "denoised_normalized_rgb.tif"

        print(f"[LOAD] {dataset_group} | {dataset_name}")
        print(f"       {zarr_dir}")

        arr_csyx, meta = load_ome_zarr_volume_csyx(
            zarr_dir,
            level=level,
            time_if_both=0,
            z_if_both=0,
            channels=None,
            prefer=("z", "t"),
            as_numpy=True,
        )

        print(f"       shape: {arr_csyx.shape}")
        print(f"       dtype: {arr_csyx.dtype}")
        print(f"       min/max raw: {arr_csyx.min()} / {arr_csyx.max()}")
        print(f"       slice axis used: {meta.get('slice_axis_used')}")

        copy_original_raw_rgb_if_exists(
            dataset_group=dataset_group,
            dataset_name=dataset_name,
            dataset_output_dir=dataset_output_dir,
        )

        arr_norm = normalize_to_original_dtype_range_per_image(
            arr_csyx,
            p_low=NORMALIZE_LOW_PERCENTILE,
            p_high=NORMALIZE_HIGH_PERCENTILE,
        )

        norm_rgb = make_rgb_csyx(
            arr_norm,
            dataset_group=dataset_group,
        )

        tif.imwrite(
            denoised_norm_rgb_out,
            norm_rgb,
            photometric="rgb",
            metadata={"axes": "SYXC"},
        )

        print(f"       saved denoised normalized RGB: {denoised_norm_rgb_out}")
        print("")


def export_all_denoised_omezarr_to_tiff() -> None:
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_group, base_dir in BASE_DIRS.items():
        export_omezarr_folder_to_grouped_normalized_colored_tiffs(
            dataset_group=dataset_group,
            base_dir=base_dir,
            output_root=OUTPUT_ROOT,
            level=LEVEL,
        )


if __name__ == "__main__":
    export_all_denoised_omezarr_to_tiff()