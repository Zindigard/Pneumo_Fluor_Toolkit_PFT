from __future__ import annotations

import numpy as np
import pytest

from PFT.core_prog_parts import decoder_omezar as dec


def test_axis_inference_and_normalization():
    assert dec.infer_axes_from_ndim(2) == "yx"
    assert dec.infer_axes_from_ndim(3) == "cyx"
    assert dec.infer_axes_from_ndim(5) == "tczyx"
    assert dec.infer_axes_from_ndim(7) == "unknown"
    assert dec.normalize_axes(" CZYX ") == "czyx"


def test_select_index_removes_axis_and_checks_bounds():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out, axes = dec.select_index_along_axis(arr, "cyx", "c", 1)
    assert axes == "yx"
    assert np.array_equal(out, arr[1])
    with pytest.raises(IndexError):
        dec.select_index_along_axis(arr, "cyx", "c", 2)


def test_move_yx_to_last_and_ensure_czyx():
    arr = np.zeros((5, 7, 2), dtype=np.uint8)  # Y X C
    moved, axes = dec.move_yx_to_last(arr, "yxc")
    assert axes == "cyx"
    assert moved.shape == (2, 5, 7)

    yx = np.zeros((5, 7), dtype=np.uint8)
    czyx, axes = dec.ensure_czyx(yx, "yx")
    assert axes == "czyx"
    assert czyx.shape == (1, 1, 5, 7)

    with pytest.raises(ValueError):
        dec.ensure_czyx(np.zeros((2, 3, 4)), "tyx")


def test_n2v_stack_and_percentile_normalization():
    arr = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    stack = dec.to_n2v_stack(arr, "cyx")
    assert stack.shape == (2, 4, 5)
    norm = dec.percentile_normalize_stack01(stack, 0, 100)
    assert norm.dtype == np.float32
    assert float(norm.min()) == 0.0
    assert float(norm.max()) == 1.0

    const = dec.percentile_normalize_stack01(np.ones((1, 3, 3), dtype=np.float32))
    assert np.count_nonzero(const) == 0


def test_get_z_slice_cyx():
    vol = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    sl = dec.get_z_slice_cyx(vol, 1)
    assert sl.shape == (2, 4, 5)
    assert np.array_equal(sl, vol[:, 1])


def test_psf_config_parser_and_required_keys(tmp_path):
    cfg_path = tmp_path / "psf.txt"
    cfg_path.write_text("# comment\nNA = 1.4\nLambda=520\nbadline\n", encoding="utf-8")
    cfg = dec.parse_psfgenerator_config(cfg_path)
    assert cfg["NA"] == "1.4"
    assert cfg["Lambda"] == "520"
    required = dec.psfgenerator_required_keys()
    assert isinstance(required, (list, tuple, set))
    ok, missing = dec.check_config_has_required({k: "1" for k in required})
    assert ok is True
    assert missing == []
