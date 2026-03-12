from __future__ import annotations
import numpy as np
import pytest
pytest.importorskip("ome_zarr")
pytest.importorskip("zarr")
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.decoder_omezar import ome_zarr_to_n2v_2d_stack

def test_decode_returns_n2v_shape(tmp_path, arr_cyx, meta_2ch):
    zdir = tmp_path / "image.ome.zarr"
    save_ome_zarr(zdir, arr_cyx, axes="cyx", meta=meta_2ch, overwrite=True)

    x = ome_zarr_to_n2v_2d_stack(zdir, channel=0, normalize="percentile")
    assert x.ndim == 4
    assert x.shape[-1] == 1
    assert x.shape[1:3] == (64, 64)
    assert x.dtype == np.float32
    assert x.min() >= 0.0
    assert x.max() <= 1.0