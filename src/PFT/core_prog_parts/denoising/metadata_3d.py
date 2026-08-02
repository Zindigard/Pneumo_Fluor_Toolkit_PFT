"""
Metadata helpers shared by the 3D restoration workflow.

The functions in this module read OME-NGFF and PFT metadata without loading a
complete image volume. Channel wavelengths are obtained from per-channel CZI
metadata whenever possible. A documented 405/488/561 nm fallback is used only
when wavelength metadata are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import zarr

from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare

DEFAULT_FALLBACK_WAVELENGTHS_NM: tuple[float, ...] = (405.0, 488.0, 561.0)


@dataclass(frozen=True)
class ChannelOptics:
    """Resolved optical information for one OME-Zarr channel."""

    index: int
    name: str
    wavelength_nm: float
    wavelength_source: str


def _as_float(value: Any) -> float | None:
    """Return as float for the supplied inputs.

    Args:
        value (Any): Value to validate, transform, store, or forward.

    Returns:
        float | None: Computed numerical result.

    Example:
        >>> result = _as_float(value=...)
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _channel_info_from_root(root: zarr.Group) -> list[dict[str, Any]]:
    """Return channel info from root for the supplied inputs.

    Args:
        root (zarr.Group): Root directory used to resolve relative project paths.

    Returns:
        list[dict[str, Any]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _channel_info_from_root(root=Path("path/to/resource"))
    """
    pft_meta = root.attrs.get("pft_meta")
    if isinstance(pft_meta, dict):
        info = pft_meta.get("channel_info")
        if isinstance(info, list):
            return [item for item in info if isinstance(item, dict)]
    return []


def _metadata_wavelength(item: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Return the best emission-like wavelength stored in one channel record.

    Args:
        item (Mapping[str, Any]): Text value specifying item.

    Returns:
        tuple[float | None, str | None]: Collection containing the generated or selected values.

    Example:
        >>> result = _metadata_wavelength(item="item")
    """
    preferred_keys = (
        "emission_nm",
        "emission_wavelength_nm",
        "emission_wavelength",
        "lambda_em_nm",
        "wavelength_nm",
        "wavelength",
        "excitation_nm",
        "excitation_wavelength_nm",
    )
    for key in preferred_keys:
        value = _as_float(item.get(key))
        if value is not None:
            return value, f"pft_meta.channel_info.{key}"
    return None, None


def _omero_wavelengths(root: zarr.Group) -> list[tuple[float | None, str | None]]:
    """Return omero wavelengths for the supplied inputs.

    Args:
        root (zarr.Group): Root directory used to resolve relative project paths.

    Returns:
        list[tuple[float | None, str | None]]: Collection containing the generated or selected values.

    Example:
        >>> result = _omero_wavelengths(root=Path("path/to/resource"))
    """
    omero = root.attrs.get("omero")
    channels = omero.get("channels") if isinstance(omero, dict) else None
    output: list[tuple[float | None, str | None]] = []
    if not isinstance(channels, list):
        return output
    for channel in channels:
        if not isinstance(channel, dict):
            output.append((None, None))
            continue
        found = None
        source = None
        for key in ("emission_wavelength", "emission_nm", "wavelength", "excitation_wavelength"):
            found = _as_float(channel.get(key))
            if found is not None:
                source = f"omero.channels.{key}"
                break
        output.append((found, source))
    return output


def _fallback_for_channel(name: str, index: int, fallback: Sequence[float]) -> tuple[float, str]:
    """Return fallback for channel for the supplied inputs.

    Args:
        name (str): Name used to identify the current object, resource, or output.
        index (int): Zero-based index of the selected element.
        fallback (Sequence[float]): Numerical value controlling fallback.

    Returns:
        tuple[float, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _fallback_for_channel(
        ...     name="name",
        ...     index=1,
        ...     fallback=0.5,
        ... )
    """
    upper = name.upper()
    name_rules = (("T1", 405.0), ("405", 405.0), ("T2", 488.0), ("488", 488.0),
                  ("T3", 561.0), ("561", 561.0), ("568", 561.0))
    for token, value in name_rules:
        if token in upper:
            return value, f"fallback channel-name rule ({token})"
    if index < len(fallback):
        return float(fallback[index]), f"fallback channel-index {index}"
    raise ValueError(
        f"No wavelength metadata or fallback exists for channel {index} ({name!r}). "
        "Provide an explicit wavelength mapping."
    )


def resolve_channel_optics(
    zarr_dir: str | Path,
    *,
    level: int = 0,
    explicit_wavelength_nm: Mapping[str, float] | None = None,
    fallback_wavelengths_nm: Sequence[float] = DEFAULT_FALLBACK_WAVELENGTHS_NM,
) -> list[ChannelOptics]:
    """Resolve channel names and wavelengths from OME-Zarr metadata.

    Resolution order is explicit mapping, ``pft_meta.channel_info``, OME/OMERO
    channel metadata, then the documented fallback rules.

    Args:
        zarr_dir (str | Path): Directory used for Zarr.
        level (int): Numerical value controlling level. Defaults to ``0``.
        explicit_wavelength_nm (Mapping[str, float] | None): Text value specifying explicit wavelength nm. ``None`` selects the function's default behavior.
        fallback_wavelengths_nm (Sequence[float]): Numerical value controlling fallback wavelengths nm. Defaults to ``DEFAULT_FALLBACK_WAVELENGTHS_NM``.

    Returns:
        list[ChannelOptics]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_channel_optics(zarr_dir=Path("path/to/resource"))
    """
    zarr_dir = Path(zarr_dir)
    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    shape = tuple(meta.get("shape") or ())
    axes = str(meta.get("axes") or "").lower()
    if "c" not in axes:
        raise ValueError(f"OME-Zarr has no channel axis: axes={axes!r}")
    channel_count = int(shape[axes.index("c")])

    root = zarr.open_group(str(zarr_dir), mode="r")
    names = meta.get("channel_names")
    if not isinstance(names, list) or len(names) < channel_count:
        names = [f"channel_{index}" for index in range(channel_count)]
    else:
        names = [str(value) for value in names[:channel_count]]

    channel_info = _channel_info_from_root(root)
    omero_values = _omero_wavelengths(root)
    resolved: list[ChannelOptics] = []
    for index, name in enumerate(names):
        value: float | None = None
        source: str | None = None
        if explicit_wavelength_nm is not None and name in explicit_wavelength_nm:
            value = _as_float(explicit_wavelength_nm[name])
            source = "explicit mapping"
        if value is None and index < len(channel_info):
            value, source = _metadata_wavelength(channel_info[index])
        if value is None and index < len(omero_values):
            value, source = omero_values[index]
        if value is None:
            value, source = _fallback_for_channel(name, index, fallback_wavelengths_nm)
        resolved.append(ChannelOptics(index, name, float(value), str(source)))
    return resolved


def coordinate_scale_for_level(zarr_dir: str | Path, *, level: int) -> list[float]:
    """Return the axis-aligned physical scale vector for a selected level.

    Args:
        zarr_dir (str | Path): Directory used for Zarr.
        level (int): Numerical value controlling level.

    Returns:
        list[float]: Collection containing the generated or selected values.

    Example:
        >>> result = coordinate_scale_for_level(zarr_dir=Path("path/to/resource"), level=1)
    """
    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    axes = str(meta.get("axes") or "czyx").lower()
    voxel = meta.get("voxel_size_um") or {}
    values = {"t": 1.0, "c": 1.0, "z": voxel.get("z"), "y": voxel.get("y"), "x": voxel.get("x")}
    result: list[float] = []
    for axis in axes:
        value = values.get(axis, 1.0)
        result.append(float(value) if value is not None else 1.0)
    return result


def copyable_root_metadata(zarr_dir: str | Path) -> dict[str, Any]:
    """Return all source root metadata that can be retained in a derived store.

    The output OME-Zarr must define its own ``multiscales`` metadata and its own
    processing record. Every other source root attribute is copied. The original
    multiscale and processing attributes are retained separately under
    ``pft_source_multiscales`` and ``pft_source_processing`` so no source metadata
    are lost during deconvolution.

    Args:
        zarr_dir (str | Path): Directory used for Zarr.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = copyable_root_metadata(zarr_dir=Path("path/to/resource"))
    """
    root = zarr.open_group(str(zarr_dir), mode="r")
    retained: dict[str, Any] = {}
    for key in root.attrs.keys():
        value = root.attrs.get(key)
        if key == "multiscales":
            retained["pft_source_multiscales"] = value
        elif key == "pft_processing":
            retained["pft_source_processing"] = value
        else:
            retained[key] = value
    retained["pft_source_root_metadata_preserved"] = True
    return retained


__all__ = [
    "ChannelOptics",
    "DEFAULT_FALLBACK_WAVELENGTHS_NM",
    "coordinate_scale_for_level",
    "copyable_root_metadata",
    "resolve_channel_optics",
]
