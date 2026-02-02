from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pytest

@dataclass
class FakeMeta:
    source_path: str = "dummy.czi"
    axes: str = "CYX"
    header_shape: tuple[int, ...] | None = None
    squeezed_shape: tuple[int, ...] | None = None
    squeezed_ndim: int | None = None
    dtype: str | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    pixel_size_um_x: float | None = 0.1
    pixel_size_um_y: float | None = 0.1
    pixel_size_um_z: float | None = 0.3
    channel_names: list[str] | None = None
    raw_xml_available: bool = True

@pytest.fixture
def arr_cyx() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.integers(0, 2**12, size=(2, 64, 64))).astype(np.uint16)

@pytest.fixture
def meta_2ch() -> FakeMeta:
    return FakeMeta(channel_names=["WGA", "DAPI"])
