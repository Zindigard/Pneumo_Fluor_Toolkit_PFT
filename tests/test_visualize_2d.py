from __future__ import annotations
import numpy as np
from PFT.core_prog_parts import visualize_2d

def test_rgb_wga_dapi_shape_and_range(arr_cyx):
    rgb = visualize_2d.rgb_wga_dapi(arr_cyx, wga_ch=0, dapi_ch=1)
    assert rgb.shape == (64, 64, 3)
    assert np.issubdtype(rgb.dtype, np.floating)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0