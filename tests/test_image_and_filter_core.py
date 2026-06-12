from __future__ import annotations

import numpy as np
import pytest

from PFT.core_prog_parts import BM3D
from PFT.core_prog_parts import floodfill_algorythm as ff
from PFT.core_prog_parts import image_utils as iu
from PFT.core_prog_parts import notch_filter as nf
from PFT.core_prog_parts import thresholding as th


def test_image_normalization_and_rgb_helpers():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    norm = iu.normalize01_percentile(x, 0, 100)
    assert norm.shape == x.shape
    assert norm.dtype == np.float32
    assert np.isclose(norm.min(), 0)
    assert np.isclose(norm.max(), 1)

    assert iu.to_uint8_minmax(x).dtype == np.uint8
    blue = np.ones((8, 9), dtype=np.float32)
    green = np.zeros((8, 9), dtype=np.float32)
    rgb = iu.rgb_wga_dapi_norm(blue, green)
    assert rgb.shape == (8, 9, 3)
    assert np.all((0 <= rgb) & (rgb <= 1))


def test_notch_mask_and_filter_preserve_shape_and_mean():
    params = nf.NotchParams(angles_deg=(0.0, 90.0), half_width_deg=5.0, depth=0.8)
    mask = nf.build_wedge_mask((64, 64), params)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0

    img = np.random.default_rng(1).normal(size=(64, 64)).astype(np.float32)
    out, used = nf.apply_notch_filter_2d(img, params)
    assert out.shape == img.shape
    assert used.shape == img.shape
    assert np.isclose(out.mean(), img.mean(), atol=1e-5)


def test_flood_fill_and_mask_helpers(tmp_path):
    img = np.zeros((9, 9), dtype=np.float32)
    img[2:7, 2:7] = 10
    result = ff.flood_mask(img, (4, 4), tolerance=0.0)
    assert result.filled_size == 25
    assert result.seed_value == 10.0

    merged = ff.merge_flood_masks(img, [(0, 0), (4, 4)], tolerance=0.0)
    assert merged.shape == img.shape
    assert merged.dtype == bool
    u8 = ff.mask_to_uint8(result.mask, true_value=200)
    assert u8.dtype == np.uint8
    assert set(np.unique(u8)) <= {0, 200}
    applied = ff.apply_mask(img, result.mask, background_value=-1)
    assert applied[0, 0] == -1
    assert applied[4, 4] == 10
    assert ff.default_floodfill_results_dir(tmp_path).is_dir()

    with pytest.raises(ValueError):
        ff.flood_mask(img, (20, 1))


def test_bm3d_normalization_roundtrip():
    x = np.arange(25, dtype=np.uint16).reshape(5, 5)
    y, inv = BM3D._normalize_to_01(x, "minmax", 1.0, 99.8)
    back = BM3D._denormalize_from_01(y, inv)
    assert y.dtype == np.float32
    assert np.all((0 <= y) & (y <= 1))
    assert np.allclose(back, x, atol=1)


def test_deconvolution_preparation_helpers(monkeypatch):
    import importlib
    import sys
    import types
    monkeypatch.setitem(sys.modules, "zarr", types.ModuleType("zarr"))
    mod = importlib.import_module("PFT.core_prog_parts.deconvolution_no_fuji")
    psf = np.ones((3, 5, 5), dtype=np.float32)
    norm = mod._normalize_psf(psf)
    assert np.isclose(norm.sum(), 1.0)
    with pytest.raises(ValueError):
        mod._normalize_psf(np.zeros((3, 3, 3), dtype=np.float32))

    image = np.array([[[-2, 1], [2, 3]]], dtype=np.float32)
    prepared = mod._prepare_image_for_rl(image, background=1.0)
    assert prepared.dtype == np.float32
    assert prepared.min() >= 0


def test_thresholding_base_intensity_and_classification():
    img = np.zeros((2, 6, 7), dtype=np.float32)
    img[0] = 1.0
    img[1] = 3.0
    base = th.build_base_intensity(img)
    assert base.shape == img.shape
    low, mid, high = th.classify_base_intensity(base, t_low=0.2, t_high=0.8)
    assert low.shape == base.shape
    assert mid.shape == base.shape
    assert high.shape == base.shape
    assert np.all((low.astype(int) + mid.astype(int) + high.astype(int)) == 1)
