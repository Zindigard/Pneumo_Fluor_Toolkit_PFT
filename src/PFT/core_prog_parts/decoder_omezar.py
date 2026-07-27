from __future__ import annotations
from pathlib import Path
from typing import Any, Tuple, Optional, Sequence
import numpy as np
"Loads and interprets OME-Zarr image data into arrays and metadata dictionaries. Used when we are working with networks."

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


def _move_axis(arr: Any, axes: str, letter: str, dest: int) -> tuple[Any, str]:
    """
    Move an axis identified by letter to a specific position (dest index).
    """
    axes = normalize_axes(axes)
    letter = letter.lower()
    if letter not in axes:
        return arr, axes

    src = axes.index(letter)
    if src == dest:
        return arr, axes

    perm = list(range(len(axes)))
    perm.pop(src)
    perm.insert(dest, src)

    arr2 = arr.transpose(perm)
    axes2 = "".join(axes[i] for i in perm)
    return arr2, axes2


def ensure_czyx(arr: Any, axes: str) -> tuple[Any, str]:
    """
    Return data in axis order 'czyx' (channel, z, y, x).
    Adds missing C/Z as singleton dims if absent.
    """
    axes = normalize_axes(axes)
    if axes == "unknown":
        raise ValueError("Cannot ensure czyx because axes='unknown'.")

    # Move yx last first
    arr, axes = move_yx_to_last(arr, axes)

    if "c" not in axes:
        arr = np.asarray(arr)[None, ...]
        axes = "c" + axes
    if "z" not in axes:
        arr = np.asarray(arr)
        arr = arr.reshape((arr.shape[0], 1) + arr.shape[1:])
        axes = axes[0] + "z" + axes[1:]

    arr, axes = _move_axis(arr, axes, "c", 0)
    arr, axes = _move_axis(arr, axes, "z", 1)

    arr, axes = move_yx_to_last(arr, axes)

    extra = [a for a in axes if a not in "czyx"]
    if extra:
        raise ValueError(f"ensure_czyx expects only axes subset of 'czyx'. Got axes='{axes}' extra={extra}")

    return arr, axes


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
    - array can be dask-like unless as_numpy=True.
    - axes is a string like 'tczyx', 'czyx', 'cyx', ...
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

def extract_ome_zarr_meta_for_compare(zarr_dir: str | Path, *, level: int = 0) -> dict[str, object]:
    """Helper function used by this module."""
    zarr_dir = Path(zarr_dir)

    try:
        import zarr
    except Exception as e:
        raise ImportError("Need `zarr` to read OME-Zarr metadata. Install: pip install zarr") from e

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
        # Fallbacks (best effort)
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
        """Internal helper used by this module."""
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
        "axes": normalize_axes(axes_str),
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

def _permute_to_match_shape(arr: Any, target_shape: tuple[int, ...]) -> Any:
    """
    If arr.shape is a permutation of target_shape, transpose arr to match target_shape.
    """
    src_shape = tuple(getattr(arr, "shape", ()))
    if len(src_shape) != len(target_shape):
        raise ValueError(f"Cannot permute: src ndim={len(src_shape)} target ndim={len(target_shape)}")

    used: set[int] = set()
    perm: list[int] = []
    for ts in target_shape:
        candidates = [i for i, ss in enumerate(src_shape) if ss == ts and i not in used]
        if not candidates:
            raise ValueError(f"Cannot permute: no axis with size {ts} in src_shape={src_shape}")
        i = candidates[0]
        used.add(i)
        perm.append(i)

    return arr.transpose(perm)


def _align_loaded_array_to_metadata(arr: Any, axes: str, meta: dict) -> tuple[Any, str]:
    """
    Force loaded arr to match OME-Zarr metadata (axes + stored shape) if it's just permuted.
    """
    expected_axes = meta.get("axes")
    expected_shape = meta.get("shape")

    if not (isinstance(expected_axes, str) and isinstance(expected_shape, tuple)):
        return arr, axes

    expected_axes = normalize_axes(expected_axes)
    src_shape = tuple(getattr(arr, "shape", ()))

    if src_shape == expected_shape:
        return arr, expected_axes

    if len(src_shape) == len(expected_shape) and sorted(src_shape) == sorted(expected_shape):
        arr2 = _permute_to_match_shape(arr, expected_shape)
        return arr2, expected_axes

    return arr, axes


def load_ome_zarr_3d_czyx(
    zarr_dir: str | Path,
    *,
    level: int = 0,
    time: int | None = 0,
    channels: Sequence[int] | None = None,
    as_numpy: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Load an OME-Zarr volume and return array in (C, Z, Y, X).

    Robust: aligns loaded array to .zattrs/.zarray metadata if the reader returns a permuted layout.
    """
    zarr_dir = Path(zarr_dir)

    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)

    arr, axes = load_ome_zarr(zarr_dir, level=level, as_numpy=False)
    axes = normalize_axes(axes)
    if axes == "unknown":
        raise ValueError(f"Axes could not be inferred for: {zarr_dir}")

    arr, axes = _align_loaded_array_to_metadata(arr, axes, meta)

    if "t" in axes:
        if time is None:
            raise ValueError("This loader requires selecting a single time. Pass time=int (e.g. 0).")
        arr, axes = select_index_along_axis(arr, axes, "t", int(time))

    # yx last before channel selection
    arr, axes = move_yx_to_last(arr, axes)

    # select channels
    if channels is not None:
        channels = list(map(int, channels))
        if "c" not in axes:
            raise ValueError(f"Requested channels={channels} but data has no 'c' axis. axes='{axes}'")
        c_i = axes.index("c")
        sl = [slice(None)] * len(getattr(arr, "shape", ()))
        sl[c_i] = channels
        arr = np.asarray(arr)[tuple(sl)]

    if as_numpy:
        arr = np.asarray(arr)

    # canonical order
    arr, axes = ensure_czyx(arr, axes)
    if axes != "czyx":
        raise RuntimeError(f"Internal error: expected axes='czyx' but got '{axes}'")

    return np.asarray(arr), "czyx"


def get_z_slice_cyx(vol_czyx: np.ndarray, z: int) -> np.ndarray:
    """
    Convenience: from volume (C,Z,Y,X) return slice (C,Y,X) at Z=z.
    """
    if vol_czyx.ndim != 4:
        raise ValueError(f"Expected vol_czyx.ndim==4 (C,Z,Y,X). Got shape={vol_czyx.shape}")
    if not (0 <= z < vol_czyx.shape[1]):
        raise IndexError(f"z={z} out of bounds for Z={vol_czyx.shape[1]}")
    return vol_czyx[:, z, :, :]


def load_ome_zarr_volume_csyx(
    zarr_dir: str | Path,
    *,
    level: int = 0,
    time_if_both: int = 0,
    z_if_both: int = 0,
    channels: Sequence[int] | None = None,
    prefer: tuple[str, str] = ("z", "t"),
    as_numpy: bool = True,
) -> tuple[np.ndarray, dict]:
    """Load data and return the processed result."""
    arr, axes = load_ome_zarr(zarr_dir, level=level, as_numpy=False)
    axes = normalize_axes(axes)
    if axes == "unknown":
        raise ValueError(f"Axes could not be inferred for: {zarr_dir}")

    def _ax_size(letter: str) -> int:
        """Internal helper used by this module."""
        if letter not in axes:
            return 1
        return int(getattr(arr, "shape")[axes.index(letter)])

    has_t = "t" in axes
    has_z = "z" in axes
    t_len = _ax_size("t") if has_t else 1
    z_len = _ax_size("z") if has_z else 1

    slice_axis = "none"
    if ("z" in prefer) and has_z and z_len > 1:
        slice_axis = "z"
    elif ("t" in prefer) and has_t and t_len > 1:
        slice_axis = "t"
    else:
        if has_z and z_len > 1:
            slice_axis = "z"
        elif has_t and t_len > 1:
            slice_axis = "t"
        else:
            slice_axis = "none"

    if has_t and has_z:
        if slice_axis == "z":
            arr, axes = select_index_along_axis(arr, axes, "t", int(time_if_both))
        elif slice_axis == "t":
            arr, axes = select_index_along_axis(arr, axes, "z", int(z_if_both))
        else:
            arr, axes = select_index_along_axis(arr, axes, "t", int(time_if_both))
            arr, axes = select_index_along_axis(arr, axes, "z", int(z_if_both))

    arr, axes = move_yx_to_last(arr, axes)

    if channels is not None:
        channels = list(map(int, channels))
        if "c" not in axes:
            raise ValueError(f"Requested channels={channels} but data has no 'c' axis. axes='{axes}'")
        c_i = axes.index("c")
        arr = np.asarray(arr)
        sl = [slice(None)] * arr.ndim
        sl[c_i] = channels
        arr = arr[tuple(sl)]

    if as_numpy:
        arr = np.asarray(arr)

    if "c" not in axes:
        arr = arr[None, ...]
        axes = "c" + axes

    if slice_axis == "z":
        if "z" not in axes:
            arr = arr.reshape((arr.shape[0], 1) + arr.shape[1:])
            axes = axes[0] + "z" + axes[1:]
        s_letter = "z"
    elif slice_axis == "t":
        if "t" not in axes:
            arr = arr.reshape((arr.shape[0], 1) + arr.shape[1:])
            axes = axes[0] + "t" + axes[1:]
        s_letter = "t"
    else:
        arr = arr.reshape((arr.shape[0], 1) + arr.shape[1:])
        axes = axes[0] + "z" + axes[1:]
        s_letter = "z"

    # permute to (C, S, Y, X)
    c_i = axes.index("c")
    s_i = axes.index(s_letter)
    y_i = axes.index("y")
    x_i = axes.index("x")
    arr = np.transpose(arr, [c_i, s_i, y_i, x_i])

    meta = {
        "zarr_dir": str(Path(zarr_dir)),
        "level": int(level),
        "axes_before_perm": axes,
        "slice_axis_used": slice_axis,  # 'z' or 't' or 'none'
        "slice_len": int(arr.shape[1]),
        "channel_len": int(arr.shape[0]),
    }
    return arr, meta


def ome_zarr_to_n2v_2d_stack(
    zarr_dir: str | Path,
    *,
    channel: int = 0,
    time: int | None = None,
    z: int | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Convert OME-Zarr into a NumPy stack shaped (N, Y, X, 1) for 2D N2V."""
    arr, axes = load_ome_zarr(zarr_dir, as_numpy=False)

    axes = normalize_axes(axes)
    if axes == "unknown":
        raise ValueError(f"Axes could not be inferred for: {zarr_dir}")

    arr, axes = select_index_along_axis(arr, axes, "c", channel)
    if time is not None:
        arr, axes = select_index_along_axis(arr, axes, "t", time)
    if z is not None:
        arr, axes = select_index_along_axis(arr, axes, "z", z)

    arr, axes = move_yx_to_last(arr, axes)

    arr_np = np.asarray(arr)
    stack = to_n2v_stack(arr_np, axes)

    if normalize == "percentile":
        stack = percentile_normalize_stack01(stack, p_low=1.0, p_high=99.8)
    elif normalize is not None:
        raise ValueError("normalize must be None or 'percentile'")

    return stack[..., None]


def decode_omezarr_volume(
    zarr_dir: str | Path,
    *,
    level: int = 0,
    time: int | None = 0,
    channels: Sequence[int] | None = None,
    as_float32: bool = False,
    print_info: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Canonical OME-Zarr decoder for 3D volumes.

    Returns
    -------
    vol_czyx : np.ndarray
        Array with shape (C, Z, Y, X).
    meta : dict
        Metadata from OME-Zarr attrs + level array info.
    """
    zarr_dir = Path(zarr_dir)

    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)

    vol, axes = load_ome_zarr_3d_czyx(
        zarr_dir,
        level=level,
        time=time,
        channels=channels,
        as_numpy=True,
    )

    if as_float32:
        vol = vol.astype(np.float32, copy=False)

    if vol.ndim != 4 or axes != "czyx":
        raise RuntimeError(f"Expected (C,Z,Y,X) with axes='czyx'. Got shape={vol.shape}, axes='{axes}'")

    if print_info:
        print("\n========= OME-ZARR DECODE INFO ==========")
        print(f"path: {zarr_dir}")
        print(f"level: {level}")
        print(f"selected time: {time if isinstance(meta.get('axes'), str) and 't' in str(meta.get('axes')) else 'N/A'}")
        print(f"selected channels: {list(channels) if channels is not None else 'ALL'}")
        print("")
        print("-- from multiscales/.zattrs --")
        print(f"axes: {meta.get('axes')}")
        print(f"voxel_size_um: {meta.get('voxel_size_um')}")
        print(f"channel_names: {meta.get('channel_names')}")
        print(f"array_path (level): {meta.get('array_path')}")
        print("")
        print("-- from level array (.zarray-like) --")
        print(f"stored shape: {meta.get('shape')}")
        print(f"stored dtype: {meta.get('dtype')}")
        print(f"stored chunks: {meta.get('chunks')}")
        print("")
        print("-- returned by decoder --")
        print(f"returned axes: {axes}")
        print(f"returned shape (C,Z,Y,X): {vol.shape}")
        print(f"returned dtype: {vol.dtype}")
        print("=========================================\n")

    meta_out = dict(meta)
    meta_out["decoded_axes"] = axes
    meta_out["decoded_shape_czyx"] = tuple(vol.shape)
    meta_out["decoded_dtype"] = str(vol.dtype)
    return vol, meta_out

def parse_psfgenerator_config(path: str | Path) -> dict[str, str]:
    """Parse input text or metadata into a structured form."""
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
    """Helper function used by this module."""
    return ["Lambda", "NA", "NX", "NY", "NZ", "ResAxial", "ResLateral", "Type"]


def check_config_has_required(cfg: dict[str, str]) -> tuple[bool, list[str]]:
    """Helper function used by this module."""
    req = psfgenerator_required_keys()
    missing = [k for k in req if k not in cfg]
    return (len(missing) == 0), missing