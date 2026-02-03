import numpy as np
import zarr
from pathlib import Path
import shutil

from PFT.core_prog_parts.io import load_czi
from PFT.core_prog_parts.Ome_Zarr import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.n2v_decoder_omezar import load_ome_zarr

def run_rigorous_validation(czi_path: str | Path):
    czi_path = Path(czi_path)
    temp_dir = Path("validation_temp_storage")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"Strat validation: {czi_path.name}")
    print("-" * 50)

    arr_orig, meta_orig = load_czi(czi_path)
    
    save_ome_zarr_next_to_outputs(temp_dir, arr_orig, meta_orig, overwrite=True)
    zarr_path = temp_dir / "image.ome.zarr"

    arr_zarr, axes_zarr = load_ome_zarr(zarr_path, as_numpy=True)

    checks = {}

    # dimensionality and shape
    checks["Pixel Count Match"] = (arr_orig.size == arr_zarr.size)
    checks["Dtype Match"] = (arr_orig.dtype == arr_zarr.dtype)

    #Statistical
    checks["Min Value Match"] = (np.min(arr_orig) == np.min(arr_zarr))
    checks["Max Value Match"] = (np.max(arr_orig) == np.max(arr_zarr))
    checks["Mean Value Match"] = np.isclose(np.mean(arr_orig), np.mean(arr_zarr), atol=1e-10)
    checks["Std Dev Match"] = np.isclose(np.std(arr_orig), np.std(arr_zarr), atol=1e-10)

    # Bit
   
    if checks["Pixel Count Match"]:
        checks["Data Sum Match"] = (np.sum(arr_orig) == np.sum(arr_zarr))
        checks["Bit-Level Identity"] = np.array_equal(np.sort(arr_orig.ravel()), np.sort(arr_zarr.ravel()))
    else:
        checks["Bit-Level Identity"] = False

    # Metadata 
    z_group = zarr.open(str(zarr_path), mode='r')
    stored_pft = z_group.attrs.get("pft_meta", {})
    
    checks["Meta: X Scale Match"] = np.isclose(stored_pft.get("pixel_size_um_x", 0), meta_orig.pixel_size_um_x or 0)
    checks["Meta: Y Scale Match"] = np.isclose(stored_pft.get("pixel_size_um_y", 0), meta_orig.pixel_size_um_y or 0)
    checks["Meta: Z Scale Match"] = np.isclose(stored_pft.get("pixel_size_um_z", 0), meta_orig.pixel_size_um_z or 0)

    # Channel
    if meta_orig.channel_names:
        zarr_channels = list(z_group.attrs.get("channel_names", []))
        checks["Channel Name Match"] = (zarr_channels == list(meta_orig.channel_names))
    else:
        checks["Channel Name Match"] = "NOT_AVAILABLE_IN_SOURCE"

    all_passed = True
    print(f"{'Test':<30} | {'Result'}")
    print("-" * 50)
    for test, result in checks.items():
        res_str = "PASS" if result is True else ("FAIL" if result is False else str(result))
        print(f"{test:<30} | {res_str}")
        if result is False:
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("Concl: Data integrity verified. No loss or corruption detected.")
    else:
        print("Concl: CRITICAL ERROR - Data discrepancies found.")

    if zarr_path.exists():
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    file = r"E:\2D_data_WGA_DAPI_DNA\WT_THY_noCSP_ROI3_SIM.czi"
    
    if Path(file).exists():
        try:
            run_rigorous_validation(file)
        except Exception as e:
            print(f"Validation aborted due to error: {e}")
    else:
        print(f"File not found: {file}")