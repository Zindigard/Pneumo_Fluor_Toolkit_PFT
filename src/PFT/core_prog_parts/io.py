from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any
import numpy as np
import czifile

"""Helper functions to read CZI files from the datasets, extract metadata and save it in a structured way for inspection."""

@dataclass(frozen=True)
class CziMeta:
    """Metadata container. 2D users can ignore all new optional fields."""
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

    # Channel names
    channel_names: list[str] | None

    # Full parsed per-channel info 
    channel_info: list[dict[str, Any]] | None

    # Full raw XML 
    raw_xml: str | None

    raw_xml_available: bool

    #  3D/SIM additions
    sample_id: str | None = None
    title: str | None = None
    creation_datetime: str | None = None
    operator: str | None = None
    microscope_system: str | None = None

    #Modality / reconstruction
    modality: str | None = None
    sim_mode: str | None = None           # "2D-SIM" / "3D-SIM" / None
    application_name: str | None = None
    application_version: str | None = None
    exported_image_type: str | None = None  # raw frames / reconstructed / unknown
    sim_settings: dict[str, str] | None = None

    # intensity
    pixel_type: str | None = None

    # Optics / detection
    objective_model: str | None = None
    objective_magnification: float | None = None
    objective_na: float | None = None
    immersion: str | None = None
    refractive_index_immersion: float | None = None
    refractive_index_sample: float | None = None
    detector_type: str | None = None
    detector_model: str | None = None



def list_czi_files(folder: str | Path) -> list[Path]:
    """List available inputs for this workflow."""
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



def _parse_float(text: str | None) -> float | None:
    """Internal helper used by this module."""
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except Exception:
        return None


def _find_text(root: ET.Element | None, xpath: str) -> str | None:
    """Internal helper used by this module."""
    if root is None:
        return None
    el = root.find(xpath)
    if el is None or el.text is None:
        return None
    s = el.text.strip()
    return s or None


def _parse_scaling_um(xml_text: str) -> tuple[float | None, float | None, float | None]:
    """
    Parse pixel sizes from XML. Convert to µm by * 1e6.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None, None, None

    def get_um(axis: str) -> float | None:
        """Helper function used by this module."""
        el = root.find(f".//Scaling/Items/Distance[@Id='{axis}']/Value")
        if el is None:
            el = root.find(f".//Scaling//Distance[@Id='{axis}']/Value")
        if el is None or el.text is None:
            return None
        try:
            meters = float(el.text.strip())
        except ValueError:
            return None
        return meters * 1e6  # meters -> µm

    return get_um("X"), get_um("Y"), get_um("Z")


def _parse_channel_names(xml_text: str) -> list[str] | None:
    """Find channel names in XML."""
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
        if n and n not in seen:
            uniq.append(n)
            seen.add(n)

    return uniq or None


def _parse_channel_info(xml_text: str) -> tuple[list[dict[str, Any]] | None, list[str] | None]:
    """Parse per-channel: label, excitation/emission, exposure, type."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None, None

    channels = root.findall(".//Information//Image//Dimensions//Channels//Channel")
    if not channels:
        return None, None

    info: list[dict[str, Any]] = []
    names: list[str] = []

    def to_nm(v: float | None) -> float | None:
        """Helper function used by this module."""
        if v is None:
            return None
        return v * 1e9 if v < 1e-3 else v

    for ch in channels:
        name = (ch.get("Name") or _find_text(ch, "Name") or "").strip() or None
        if name:
            names.append(name)

        ex = _parse_float(_find_text(ch, "ExcitationWavelength"))
        em = _parse_float(_find_text(ch, "EmissionWavelength"))
        ch_type = _find_text(ch, "ChannelType")

        exposure_raw = _parse_float(_find_text(ch, "ExposureTime"))
        exposure_s = None
        if exposure_raw is not None:
            exposure_s = exposure_raw / 1e9 if exposure_raw > 1e6 else exposure_raw

        info.append(
            {
                "name": name,
                "channel_type": ch_type,
                "excitation_nm": to_nm(ex),
                "emission_nm": to_nm(em),
                "exposure_s": exposure_s,
            }
        )

    uniq: list[str] = []
    seen: set[str] = set()
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)

    return info or None, uniq or None


def _parse_sim_settings_from_lsm_tags(xml_text: str) -> dict[str, str] | None:
    """
    Zeiss SIM settings.

    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None

    wanted_prefixes = (
        "OTFThreshold",
        "NotchFilter",
        "Apodization",
        "Sectioning",
        "Sharpness",
        "BaselineCut",
        "Destriping",
        "Optimise",
        "GratingPeriod",
        "Rotations",
        "Phases",
        "FixedOversampling",
        "OutputOversampling",
        "RI",
        "MLriImm",
        "MLriSam",
        "MLriCov",
    )

    out: dict[str, str] = {}
    for tag in root.findall(".//LsmTag"):
        txt = (tag.text or "").strip()
        if not txt:
            continue
        for line in txt.replace("\r", "").split("\n"):
            parts = [p for p in line.split("\t") if p != ""]
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            val = "\t".join(parts[1:]).strip()
            if key and val and key.startswith(wanted_prefixes):
                out.setdefault(key, val)

    return out or None


def parse_czi_3d_metadata(xml_text: str) -> dict[str, Any]:
    """
    Best-effort metadata fields.
    Returns a dict that can be merged into CziMeta.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return {}

    sample_id = _find_text(root, ".//Information/Image/Name")
    title = _find_text(root, ".//Information/Image/Title")
    creation = _find_text(root, ".//Information/Image/CreationDate")
    operator = _find_text(root, ".//Information/Image/UserName")
    microscope_system = _find_text(root, ".//Information/Instrument/Microscopes/Microscope/System")

    app_name = _find_text(root, ".//Information/Application/Name")
    app_ver = _find_text(root, ".//Information/Application/Version")

    comment = _find_text(root, ".//Information/Image/Comment")
    has_sim_block = root.find(".//Processing/StructuredIllumination") is not None
    modality = "SIM" if has_sim_block or (comment and "Structured Illumination" in comment) else None

    sim_mode = None
    if comment:
        if "Processing : 3D" in comment or "Processing: 3D" in comment:
            sim_mode = "3D-SIM"
        elif "Processing : 2D" in comment or "Processing: 2D" in comment:
            sim_mode = "2D-SIM"
    if sim_mode is None and has_sim_block:
        sim_mode = "3D-SIM"

    sim_settings = _parse_sim_settings_from_lsm_tags(xml_text)
    if comment:
        if sim_settings is None:
            sim_settings = {}
        sim_settings.setdefault("ZEN_Comment", comment)

    pixel_type = _find_text(root, ".//Information/Image/PixelType")

    obj = root.find(".//Information/Instrument/Objectives/Objective")
    objective_model = _find_text(obj, "Manufacturer/Model") if obj is not None else None
    objective_mag = _parse_float(_find_text(obj, "NominalMagnification")) if obj is not None else None
    objective_na = _parse_float(_find_text(obj, "LensNA")) if obj is not None else None
    immersion = _find_text(obj, "Immersion") if obj is not None else None

    det = root.find(".//Information/Instrument/Detectors/Detector")
    detector_type = _find_text(det, "Type") if det is not None else None
    detector_model = _find_text(det, "Manufacturer/Model") if det is not None else None

    ri_imm = _parse_float(sim_settings.get("RI") if isinstance(sim_settings, dict) else None)
    ri_sam = _parse_float(sim_settings.get("MLriSam") if isinstance(sim_settings, dict) else None)

    return {
        "sample_id": sample_id,
        "title": title,
        "creation_datetime": creation,
        "operator": operator,
        "microscope_system": microscope_system,
        "application_name": app_name,
        "application_version": app_ver,
        "modality": modality,
        "sim_mode": sim_mode,
        "exported_image_type": "unknown",
        "sim_settings": sim_settings,
        "pixel_type": pixel_type,
        "objective_model": objective_model,
        "objective_magnification": objective_mag,
        "objective_na": objective_na,
        "immersion": immersion,
        "refractive_index_immersion": ri_imm,
        "refractive_index_sample": ri_sam,
        "detector_type": detector_type,
        "detector_model": detector_model,
    }

def load_czi(path: str | Path) -> tuple[np.ndarray, CziMeta]:
    """
    Load array (squeezed) + parse metadata (pixel size + channel names).
    2D remains unchanged; 3D simply gets extra optional fields filled.
    """
    path = Path(path)
    axes, header_shape = read_czi_header(path)
    arr = read_czi_array_squeezed(path)

    xml = read_czi_xml(path)
    if xml is not None:
        sx_um, sy_um, sz_um = _parse_scaling_um(xml)
        channel_info, channel_names = _parse_channel_info(xml)
        if channel_names is None:
            channel_names = _parse_channel_names(xml)

        extra = parse_czi_3d_metadata(xml)
        xml_ok = True
    else:
        sx_um = sy_um = sz_um = None
        channel_names = None
        channel_info = None
        extra = {}
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
        channel_info=channel_info,
        raw_xml=xml,
        raw_xml_available=xml_ok,
        **{k: v for k, v in extra.items() if k in CziMeta.__dataclass_fields__},
    )
    return arr, meta


def load_czi_metadata_only(path: str | Path) -> CziMeta:
    """
    Metadata-only loader (used by 3D export if you ever want it).
    """
    path = Path(path)
    axes, header_shape = read_czi_header(path)

    xml = read_czi_xml(path)
    if xml is not None:
        sx_um, sy_um, sz_um = _parse_scaling_um(xml)
        channel_info, channel_names = _parse_channel_info(xml)
        if channel_names is None:
            channel_names = _parse_channel_names(xml)
        extra = parse_czi_3d_metadata(xml)
        xml_ok = True
    else:
        sx_um = sy_um = sz_um = None
        channel_names = None
        channel_info = None
        extra = {}
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
        channel_info=channel_info,
        raw_xml=xml,
        raw_xml_available=xml_ok,
        **{k: v for k, v in extra.items() if k in CziMeta.__dataclass_fields__},
    )
