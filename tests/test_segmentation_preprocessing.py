"""Unit tests for the segmentation normalization and metric contracts."""

from __future__ import annotations

import numpy as np

from PFT.core_prog_parts.segmentation.instance_segmentation_core import (
    instance_f1,
    semantic_dice,
    semantic_iou,
)
from PFT.core_prog_parts.segmentation.segmentation_input_core import (
    create_segmentation_input,
    select_dataset_channels,
)


def test_normalize_complete_then_apply_shared_mask_cyx() -> None:
    source = np.zeros((2, 32, 40), dtype=np.uint16)
    y, x = np.mgrid[:32, :40]
    source[0] = (100 + y * 10 + x).astype(np.uint16)
    source[1] = (500 + y * 4 + x * 3).astype(np.uint16)
    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[5:28, 7:34] = 1

    prepared, records = create_segmentation_input(
        source,
        "cyx",
        predicted_mask=mask,
        mask_axes="yx",
    )

    assert prepared.dtype == np.float32
    assert prepared.shape == source.shape
    assert 0.0 <= float(prepared.min()) <= float(prepared.max()) <= 1.0
    assert np.count_nonzero(prepared) > 0
    assert np.all(prepared[:, mask == 0] == 0)
    assert len(records) == 2


def test_shared_mask_broadcasts_to_yxc() -> None:
    source = np.arange(16 * 20 * 2, dtype=np.uint16).reshape(16, 20, 2)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[2:14, 3:18] = 1

    prepared, _ = create_segmentation_input(
        source,
        "yxc",
        predicted_mask=mask,
        mask_axes="yx",
    )

    assert prepared.shape == source.shape
    assert np.all(prepared[mask == 0] == 0)


def test_instance_metrics_for_exact_match() -> None:
    reference = np.zeros((10, 10), dtype=np.int32)
    reference[1:4, 1:4] = 1
    reference[6:9, 6:9] = 2

    assert semantic_dice(reference, reference) == 1.0
    assert semantic_iou(reference, reference) == 1.0
    assert instance_f1(reference, reference) == (1.0, 2, 0, 0)


def test_2d_time_always_selects_channel_zero() -> None:
    source = np.zeros((2, 8, 10), dtype=np.uint16)
    source[0] = 125
    source[1] = 5000

    selected, axes, metadata = select_dataset_channels(source, "cyx", "2d_time")

    assert axes == "yx"
    assert selected.shape == (8, 10)
    assert np.array_equal(selected, source[0])
    assert metadata["policy"] == "use_channel_0_only"
    assert metadata["selected_channels"] == [0]
    assert metadata["ignored_channels"] == [1]


def test_2d_time_yx_is_already_channel_zero() -> None:
    source = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9)

    selected, axes, metadata = select_dataset_channels(source, "yx", "2d_time")

    assert axes == "yx"
    assert np.array_equal(selected, source)
    assert metadata["selected_channels"] == [0]
    assert metadata["ignored_channels"] == []
