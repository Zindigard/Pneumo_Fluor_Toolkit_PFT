from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import types

# The core package imports the CZI reader at package-import time. Most unit tests
# do not read CZI files, so provide a lightweight optional-dependency stub when
# czifile is not installed.
try:
    import czifile  # type: ignore  # noqa: F401
except Exception:
    sys.modules.setdefault("czifile", types.ModuleType("czifile"))

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class FakeMeta:
    source_path: str = "dummy.czi"
    axes: str = "cyx"
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
    raw_xml: str = "<Metadata/>"


@pytest.fixture
def arr_cyx() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 2**12, size=(2, 64, 64), dtype=np.uint16)


@pytest.fixture
def meta_2ch() -> FakeMeta:
    return FakeMeta(channel_names=["WGA", "DAPI"])


@pytest.fixture
def instance_mask_2d() -> np.ndarray:
    mask = np.zeros((48, 64), dtype=np.int32)
    mask[8:18, 7:25] = 1
    mask[27:40, 35:54] = 2
    return mask


@pytest.fixture
def binary_mask_3d() -> np.ndarray:
    mask = np.zeros((3, 32, 40), dtype=np.uint8)
    mask[:, 8:20, 10:28] = 1
    return mask
