from __future__ import annotations
from pathlib import Path
from typing import Any, Tuple
import numpy as np

"""Helper functions to read OME-Zarr with ome-zarr library and convert to N2V stacks or anyother information type."""

def infer_axes_from_ndim(ndim: int | None) -> str:
    """Fallback axes inference if multiscales axes are missing."""
    if ndim == 2:
        return "yx"
    if ndim == 3:
        return "cyx"
    if ndim == 4:
        return "tcyx"
    if ndim == 5:
        return "tczyx"
    return "unknown"


def normalize_axes(axes: str) -> str:
    """Normalize axes string to lowercase."""
    return axes.strip().lower()


def select_index_along_axis(arr: Any, axes: str, letter: str, idx: int) -> tuple[Any, str]:
    """
    Select a single index along `letter` axis and remove that axis from axes string.

    """
    axes = normalize_axes(axes)
    letter = letter.lower()

    if letter not in axes:
        return arr, axes

    ax = axes.index(letter)

    shape = getattr(arr, "shape", None)
    if shape is not None:
        if ax >= len(shape):
            raise ValueError(f"Axis '{letter}' maps to ax={ax} but arr.shape={shape} for axes='{axes}'")
        if not (0 <= idx < shape[ax]):
            raise IndexError(
                f"Index {idx} out of bounds for axis '{letter}' (axis={ax}, size={shape[ax]}) "
                f"for axes='{axes}' shape={shape}"
            )

    sl = [slice(None)] * len(getattr(arr, "shape", ()))
    sl[ax] = idx
    arr2 = arr[tuple(sl)]
    axes2 = axes[:ax] + axes[ax + 1 :]
    return arr2, axes2


def move_yx_to_last(arr: Any, axes: str) -> tuple[Any, str]:
    """
    Ensure y and x are the last axes (… y x).
    """
    axes = normalize_axes(axes)
    if axes == "yx":
        return arr, axes

    if ("y" not in axes) or ("x" not in axes):
        return arr, axes

    y_i = axes.index("y")
    x_i = axes.index("x")

    if y_i == len(axes) - 2 and x_i == len(axes) - 1:
        return arr, axes

    perm = [i for i in range(len(axes)) if i not in (y_i, x_i)] + [y_i, x_i]
    arr2 = arr.transpose(perm)
    axes2 = "".join(axes[i] for i in perm)
    return arr2, axes2


def to_n2v_stack(arr_np: np.ndarray, axes: str) -> np.ndarray:
    """
    Convert array with axes ending in yx to stack (N, Y, X).
    """
    axes = normalize_axes(axes)

    if axes == "yx":
        return arr_np[None, ...]

    if axes.endswith("yx"):
        n = int(np.prod(arr_np.shape[:-2]))
        y, x = arr_np.shape[-2], arr_np.shape[-1]
        return arr_np.reshape(n, y, x)

    raise ValueError(f"Cannot convert to 2D N2V stack from axes='{axes}' shape={arr_np.shape}")


def percentile_normalize_stack01(stack: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """
    Percentile normalize each frame to [0,1] float32.
    Input stack: (N, Y, X)
    """
    stack = stack.astype(np.float32, copy=False)
    out = np.empty_like(stack, dtype=np.float32)

    for i in range(stack.shape[0]):
        frame = stack[i]
        lo = np.percentile(frame, p_low)
        hi = np.percentile(frame, p_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[i] = 0.0
        else:
            out[i] = (frame - lo) / (hi - lo)
            out[i] = np.clip(out[i], 0.0, 1.0)

    return out


def load_ome_zarr(
    zarr_dir: str | Path,
    *,
    level: int = 0,
    as_numpy: bool = False,
) -> Tuple[Any, str]:
    """
    Load OME-Zarr and return (array, axes).
    """
    zarr_dir = Path(zarr_dir)

    try:
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader
    except Exception as e:
        raise ImportError(
            "Reading OME-Zarr requires `ome-zarr`.\n"
            "Install with: pip install ome-zarr zarr numcodecs dask"
        ) from e

    loc = parse_url(str(zarr_dir), mode="r")
    reader = Reader(loc)
    nodes = list(reader())
    if not nodes:
        raise ValueError(f"No image nodes found in: {zarr_dir}")

    node = nodes[0]
    data = node.data[level]

    axes: str | None = None
    try:
        ms = node.metadata.get("multiscales", [])
        if ms and "axes" in ms[0]:
            axes_entries = ms[0]["axes"]
            axes = "".join(a["name"] if isinstance(a, dict) else str(a) for a in axes_entries)
    except Exception:
        axes = None

    if axes is None:
        axes = infer_axes_from_ndim(getattr(data, "ndim", None))

    axes = normalize_axes(axes)

    if as_numpy:
        data = np.asarray(data)

    return data, axes


def ome_zarr_to_n2v_2d_stack(
    zarr_dir: str | Path,
    *,
    channel: int = 0,
    time: int | None = None,
    z: int | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Convert OME-Zarr into a NumPy stack shaped (N, Y, X, 1) for 2D N2V.

    - If 'c' exists -> pick `channel`
    - If `time` is not None and 't' exists -> pick that time index
    - If `z` is not None and 'z' exists -> pick that z index

    Returns
    -------
    x: np.ndarray
        Shape (N, Y, X, 1), dtype float32 if normalized else original dtype promoted.
    """
    arr, axes = load_ome_zarr(zarr_dir, as_numpy=False)

    # normalize
    axes = normalize_axes(axes)
    if axes == "unknown":
        raise ValueError(f"Axes could not be inferred for: {zarr_dir}")

    # Select channel/time/z if present
    arr, axes = select_index_along_axis(arr, axes, "c", channel)
    if time is not None:
        arr, axes = select_index_along_axis(arr, axes, "t", time)
    if z is not None:
        arr, axes = select_index_along_axis(arr, axes, "z", z)
   
    arr, axes = move_yx_to_last(arr, axes)

    arr_np = np.asarray(arr)

    #(N, Y, X)
    stack = to_n2v_stack(arr_np, axes)

    if normalize == "percentile":
        stack = percentile_normalize_stack01(stack, p_low=1.0, p_high=99.8)
    elif normalize is not None:
        raise ValueError("normalize must be None or 'percentile'")

    # Return (N, Y, X, 1)
    return stack[..., None]


from typing import Optional

def extract_ome_zarr_meta_for_compare(zarr_dir: str | Path, *, level: int = 0) -> dict[str, object]:
    """
    Extract a stable subset of OME-NGFF metadata
    for comparing 3D images and checking PSFGenerator configs.

    dict
        Flat dict with keys:
        - zarr_dir, level
        - shape, ndim, dtype, chunks (if available)
        - axes (string, e.g. 'czyx' / 'tczyx')
        - voxel_size_um: {'x':..., 'y':..., 'z':...} (from multiscales scale if available)
        - channel_names (attrs/omero if available)
        - pft_meta_available (bool) + pft_meta (dict or None)
        - objective_na, objective_magnification, immersion, refractive_index_immersion, refractive_index_sample
        - modality, sim_mode
        - source_path 
    """
    zarr_dir = Path(zarr_dir)

    # We use zarr directly because it's simple for attrs + chunks/dtype
    try:
        import zarr
    except Exception as e:
        raise ImportError("Need `zarr` to read OME-Zarr. Install: pip install zarr") from e

    root = zarr.open_group(str(zarr_dir), mode="r")

    array_path = "0"
    ms = root.attrs.get("multiscales")
    if isinstance(ms, list) and ms:
        datasets = ms[0].get("datasets")
        if isinstance(datasets, list) and datasets:
            if level >= len(datasets):
                raise IndexError(f"Requested level={level} but only {len(datasets)} multiscale datasets exist.")
            p = datasets[level].get("path")
            if isinstance(p, str) and p.strip():
                array_path = p.strip()

    arr = root[array_path]

    axes_str: Optional[str] = None
    if isinstance(ms, list) and ms:
        axes = ms[0].get("axes")
        if isinstance(axes, list) and axes:
            names = []
            ok = True
            for a in axes:
                nm = a.get("name") if isinstance(a, dict) else None
                if not (isinstance(nm, str) and len(nm) == 1):
                    ok = False
                    break
                names.append(nm.lower())
            if ok:
                axes_str = "".join(names)

   
    if axes_str is None:
        if arr.ndim == 4:
            axes_str = "czyx"  
        elif arr.ndim == 5:
            axes_str = "tczyx"
        elif arr.ndim == 3:
            axes_str = "zyx"
        elif arr.ndim == 2:
            axes_str = "yx"
        else:
            axes_str = "unknown"

    voxel_um = {"x": None, "y": None, "z": None}
    if isinstance(ms, list) and ms:
        datasets = ms[0].get("datasets")
        if isinstance(datasets, list) and datasets and level < len(datasets):
            ct = datasets[level].get("coordinateTransformations")
            if isinstance(ct, list):
                for t in ct:
                    if isinstance(t, dict) and t.get("type") == "scale" and isinstance(t.get("scale"), list):
                        scale = t["scale"]
                        for ax, sc in zip(axes_str, scale):
                            if ax in ("x", "y", "z"):
                                try:
                                    voxel_um[ax] = float(sc)
                                except Exception:
                                    voxel_um[ax] = None
                        break

    channel_names = root.attrs.get("channel_names")
    if channel_names is None:
        omero = root.attrs.get("omero")
        if isinstance(omero, dict):
            chs = omero.get("channels")
            if isinstance(chs, list):
                labels = []
                for c in chs:
                    if isinstance(c, dict) and c.get("label"):
                        labels.append(c["label"])
                channel_names = labels or None

    pft_meta = root.attrs.get("pft_meta")
    def _pft_get(key: str):
        if isinstance(pft_meta, dict):
            return pft_meta.get(key)
        return None

    out: dict[str, object] = {
        "zarr_dir": str(zarr_dir),
        "level": int(level),
        "array_path": array_path,
        "shape": tuple(arr.shape),
        "ndim": int(arr.ndim),
        "dtype": str(getattr(arr, "dtype", "")),
        "chunks": getattr(arr, "chunks", None),
        "axes": axes_str,
        "voxel_size_um": voxel_um,
        "channel_names": channel_names,
        "pft_meta_available": isinstance(pft_meta, dict),
        "pft_meta": pft_meta if isinstance(pft_meta, dict) else None,
        "objective_na": _pft_get("objective_na"),
        "objective_magnification": _pft_get("objective_magnification"),
        "immersion": _pft_get("immersion"),
        "refractive_index_immersion": _pft_get("refractive_index_immersion"),
        "refractive_index_sample": _pft_get("refractive_index_sample"),
        "modality": _pft_get("modality"),
        "sim_mode": _pft_get("sim_mode"),
        "source_path": root.attrs.get("source_path") or _pft_get("source_path"),
    }
    return out


def parse_psfgenerator_config(path: str | Path) -> dict[str, str]:
    """Parse PSFGenerator 'properties-like' config file as key=value."""
    path = Path(path)
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def psfgenerator_required_keys() -> list[str]:
    """Keys we treat as required for generating a PSF compatible """
    return ["Lambda", "NA", "NX", "NY", "NZ", "ResAxial", "ResLateral", "Type"]


def check_config_has_required(cfg: dict[str, str]) -> tuple[bool, list[str]]:
    req = psfgenerator_required_keys()
    missing = [k for k in req if k not in cfg]
    return (len(missing) == 0), missing

