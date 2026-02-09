from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import czifile


@dataclass(frozen=True)
class CziMeta:
    """Minimal metadata."""
    source_path: str

    # CZI header fields (as in czifile)
    axes: str | None
    header_shape: tuple[int, ...] | None

    # array info (only available when load pixels)
    squeezed_shape: tuple[int, ...] | None
    squeezed_ndim: int | None
    dtype: str | None
    intensity_min: float | None
    intensity_max: float | None

    # Physical pixel sizes in µm
    pixel_size_um_x: float | None
    pixel_size_um_y: float | None
    pixel_size_um_z: float | None

    # Channel names from XML
    channel_names: list[str] | None

    raw_xml_available: bool


def list_czi_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = sorted(folder.glob("*.czi"))
    if not files:
        raise FileNotFoundError(f"No .czi files found in: {folder}")
    return files


def read_czi_header(path: str | Path) -> tuple[str | None, tuple[int, ...] | None]:
    """Read only CZI header fields (axes, shape)."""
    path = Path(path)
    with czifile.CziFile(str(path)) as czi:
        axes = getattr(czi, "axes", None)
        shape = getattr(czi, "shape", None)
    return axes, shape


def read_czi_array_squeezed(path: str | Path) -> np.ndarray:
    """Read CZI pixels into numpy and squeeze."""
    path = Path(path)
    with czifile.CziFile(str(path)) as czi:
        arr = czi.asarray()
    return np.squeeze(arr)


def read_czi_xml(path: str | Path) -> str | None:
    """Return raw CZI metadata XML string."""
    path = Path(path)
    try:
        with czifile.CziFile(str(path)) as czi:
            xml = czi.metadata()
        if isinstance(xml, str) and xml.strip():
            return xml
        return None
    except Exception:
        return None


def _parse_scaling_um(xml_text: str) -> tuple[float | None, float | None, float | None]:
    """
    Parse pixel sizes from XML. Convert to µm by * 1e6.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None, None, None

    def get_um(axis: str) -> float | None:
        el = root.find(f".//Scaling/Items/Distance[@Id='{axis}']/Value")
        if el is None:
            el = root.find(f".//Scaling//Distance[@Id='{axis}']/Value")  # f
        if el is None or el.text is None:
            return None
        try:
            meters = float(el.text.strip())
        except ValueError:
            return None
        return meters * 1e6

    return get_um("X"), get_um("Y"), get_um("Z")


def _parse_channel_names(xml_text: str) -> list[str] | None:
    """
    Finds channel name in xml .
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None

    names: list[str] = []
    for ch in root.findall(".//Information//Image//Dimensions//Channels//Channel"):
        nm = ch.get("Name")
        if nm:
            names.append(nm.strip())
        else:
            child = ch.find("Name")
            if child is not None and child.text:
                names.append(child.text.strip())

    # de-duplicate
    uniq: list[str] = []
    seen: set[str] = set()
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)

    return uniq or None


def load_czi(path: str | Path) -> tuple[np.ndarray, CziMeta]:
    """
    Load array (squeezed) + parse metadata (pixel size + channel names).
    """
    path = Path(path)
    axes, header_shape = read_czi_header(path)
    arr = read_czi_array_squeezed(path)

    xml = read_czi_xml(path)
    if xml is not None:
        sx_um, sy_um, sz_um = _parse_scaling_um(xml)
        channel_names = _parse_channel_names(xml)
        xml_ok = True
    else:
        sx_um = sy_um = sz_um = None
        channel_names = None
        xml_ok = False

    meta = CziMeta(
        source_path=str(path),
        axes=axes,
        header_shape=header_shape,
        squeezed_shape=tuple(arr.shape),
        squeezed_ndim=int(arr.ndim),
        dtype=str(arr.dtype),
        intensity_min=float(arr.min()) if arr.size else None,
        intensity_max=float(arr.max()) if arr.size else None,
        pixel_size_um_x=sx_um,
        pixel_size_um_y=sy_um,
        pixel_size_um_z=sz_um,
        channel_names=channel_names,
        raw_xml_available=xml_ok,
    )
    return arr, meta


def load_czi_metadata_only(path: str | Path) -> CziMeta:
    """
    Metadata-only loader only 3d.
    """
    path = Path(path)
    axes, header_shape = read_czi_header(path)

    xml = read_czi_xml(path)
    if xml is not None:
        sx_um, sy_um, sz_um = _parse_scaling_um(xml)
        channel_names = _parse_channel_names(xml)
        xml_ok = True
    else:
        sx_um = sy_um = sz_um = None
        channel_names = None
        xml_ok = False

    return CziMeta(
        source_path=str(path),
        axes=axes,
        header_shape=header_shape,
        squeezed_shape=None,
        squeezed_ndim=None,
        dtype=None,
        intensity_min=None,
        intensity_max=None,
        pixel_size_um_x=sx_um,
        pixel_size_um_y=sy_um,
        pixel_size_um_z=sz_um,
        channel_names=channel_names,
        raw_xml_available=xml_ok,
    )



def read_czi_array_raw(path: str | Path) -> tuple[np.ndarray, str | None]:
    """Read CZI pixels WITHOUT squeezing and return (arr_raw, axes_string).

    This is useful if you want to export OME-Zarr with perfect axis mapping.
    Your existing `read_czi_array_squeezed` stays untouched.
    """
    path = Path(path)
    with czifile.CziFile(str(path)) as czi:
        arr = czi.asarray()
        axes = getattr(czi, "axes", None)
    return arr, axes


def squeeze_with_axes(arr: np.ndarray, axes: str | None) -> tuple[np.ndarray, str | None]:
    """Squeeze singleton dims and drop corresponding axis letters.

    If `axes` is None or len(axes)!=arr.ndim, falls back to np.squeeze and returns axes unchanged.
    """
    if axes is None or len(axes) != arr.ndim:
        return np.squeeze(arr), axes

    keep_slices: list[slice | int] = []
    keep_axes: list[str] = []
    for letter, size in zip(axes, arr.shape):
        if size == 1:
            keep_slices.append(0)
        else:
            keep_slices.append(slice(None))
            keep_axes.append(letter)

    arr2 = arr[tuple(keep_slices)]
    return arr2, "".join(keep_axes)