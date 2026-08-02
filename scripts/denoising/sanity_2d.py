r"""Provide command-line and programmatic utilities for sanity two-dimensional data.

Examples
--------
Run the configured two-dimensional OME-Zarr sanity check:

    python scripts/denoising/sanity_2d.py
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    The lookup is based on this script's physical location and therefore does
    not depend on the current working directory. An explicit error is raised
    when the expected repository layout cannot be found.

    Args:
        start (_PFTPath | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        _PFTPath: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _pft_project_root()
    """
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root
import random
from pathlib import Path
import numpy as np
from PFT.core_prog_parts import visualization as viz
from PFT.core_prog_parts.image_utils import normalize01_percentile, rgb_wga_dapi_norm
import zarr

""" Sanity check for 2-channel OME-Zarr """

OMEZARR_PATH = Path(
    r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\training_files\2d_wga_dapi\training_data\WT_CSP_NHS_ROI2_SIM_omezarr"
)


def percentile_norm01(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Compatibility wrapper around the shared percentile normalization helper.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = percentile_norm01(x=image_array)
    """
    return normalize01_percentile(x, p_lo=p_low, p_hi=p_high)


def load_highest_res_array(zarr_path: Path) -> np.ndarray:
    """Loads highest-res dataset. Typical NGFF layout: root['0'] is the full-res array.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = load_highest_res_array(zarr_path=Path("path/to/resource"))
    """
    root = zarr.open(str(zarr_path), mode="r")

    if "0" in root:
        return np.asarray(root["0"])

    keys = list(root.array_keys())
    if keys:
        return np.asarray(root[keys[0]])

    for gk in root.group_keys():
        g = root[gk]
        ak = list(getattr(g, "array_keys", lambda: [])())
        if ak:
            return np.asarray(g[ak[0]])

    raise RuntimeError(f"Could not find any array dataset inside: {zarr_path}")


def choose_random_2d_slice(arr: np.ndarray) -> np.ndarray:
    """Returns a 3D array with channels + 2D spatial image: either (C,Y,X) or (Y,X,C).

    If arr has extra dims (T/Z), it picks random indices for those dims.

    Args:
        arr (np.ndarray): Array containing arr.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = choose_random_2d_slice(arr=image_array)
    """
    a = np.asarray(arr)

    if a.ndim < 3:
        raise ValueError(f"Expected at least 3D with channels, got shape={a.shape}")

    shape = a.shape
    small_axes = [i for i, s in enumerate(shape) if s <= 4]

    if not small_axes:
        raise ValueError(f"Cannot find a channel-like axis (<=4) in shape={shape}")

    candidate_axes = [ax for ax in small_axes if ax not in (a.ndim - 1, a.ndim - 2)]
    cax = candidate_axes[0] if candidate_axes else small_axes[0]

    idx = []
    for ax, size in enumerate(shape):
        if ax == cax:
            idx.append(slice(None))  # keep all channels
        elif ax in (a.ndim - 2, a.ndim - 1):
            idx.append(slice(None))  # keep Y,X
        else:
            idx.append(random.randrange(size))  # pick random 

    a2 = a[tuple(idx)]

    if a2.ndim != 3:
        raise ValueError(f"After slicing, expected 3D but got shape={a2.shape}")

    return a2


def split_channels(a3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract channel 0 and 1 from either (C,Y,X) or (Y,X,C).

    Args:
        a3 (np.ndarray): Array containing a3.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = split_channels(a3=image_array)
    """
    if a3.shape[0] <= 4:  # (C,Y,X)
        if a3.shape[0] < 2:
            raise ValueError(f"Need at least 2 channels, got shape={a3.shape}")
        return a3[0], a3[1]

    if a3.shape[-1] <= 4:  # (Y,X,C)
        if a3.shape[-1] < 2:
            raise ValueError(f"Need at least 2 channels, got shape={a3.shape}")
        return a3[..., 0], a3[..., 1]

    raise ValueError(f"Cannot infer channel axis from shape={a3.shape}")


def main() -> None:
    """Execute the command-line workflow and return its process exit status.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    if not OMEZARR_PATH.exists():
        raise FileNotFoundError(f"OME-Zarr not found: {OMEZARR_PATH}")

    arr = load_highest_res_array(OMEZARR_PATH)
    a3 = choose_random_2d_slice(arr)
    c0, c1 = split_channels(a3)

    c0n = percentile_norm01(c0)
    c1n = percentile_norm01(c1)

    rgb = rgb_wga_dapi_norm(c0, c1)

    viz.show_two_channel_overlay(
        c0n,
        c1n,
        rgb,
        f"{OMEZARR_PATH.name} | raw shape={arr.shape} | shown slice shape={a3.shape}",
    )


if __name__ == "__main__":
    main()

