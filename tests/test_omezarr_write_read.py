from __future__ import annotations
import numpy as np
import pytest
import zarr
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr

pytest.importorskip("ome_zarr") 

def _load_level0(zarr_dir):
    root = zarr.open_group(str(zarr_dir), mode="r")
    assert "0" in root
    return np.asarray(root["0"])

def test_omezarr_roundtrip_exact(tmp_path, arr_cyx, meta_2ch):
    out = tmp_path / "image.ome.zarr"
    save_ome_zarr(out, arr_cyx, axes="cyx", meta=meta_2ch, overwrite=True)

    arr_back = _load_level0(out)
    assert arr_back.shape == arr_cyx.shape
    assert arr_back.dtype == arr_cyx.dtype
    assert np.array_equal(arr_back, arr_cyx)