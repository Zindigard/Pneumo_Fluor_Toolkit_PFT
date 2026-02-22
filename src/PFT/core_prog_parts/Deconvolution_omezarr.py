from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.fuji_managment import find_project_root, ensure_fiji_in_project
from PFT.core_prog_parts.psf_creator import _find_java_exe
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare

from PFT.core_prog_parts.dl2 import find_dl2_jar, run_dl2_cli, write_imagej_tiff_stack


PSFModel = Literal["BW", "GL", "RW"]

DEFAULT_CHANNEL_WAVELENGTH_NM = {
    "TV1-T1-SR": 405.0,  # channel index 0 (blue)
    "TV1-T2-SR": 488.0,  # channel index 1 (green)
    "TV1-T3-SR": 561.0,  # channel index 2 (red)
}


@dataclass(frozen=True)
class DeconvRunInfo:
    in_zarr: Path
    out_zarr: Path
    out_dir: Path
    model: PSFModel
    iters: int
    background: float


def _psf_path_for(model: PSFModel, channel_name: str, wavelength_nm: float, project_root: Path) -> Path:
    psf_dir = project_root / "results" / "psf" / "generated"
    return psf_dir / f"psf_{model}_{channel_name}_Lambda{int(round(wavelength_nm))}nm.tif"


def _ensure_empty_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _write_single_scale_omezarr_like_input(
    *,
    out_zarr: Path,
    in_meta: dict,
    shape_czyx: tuple[int, int, int, int],
    dtype: np.dtype,
    chunks_czyx: tuple[int, int, int, int] | None,
) -> zarr.Array:
    """
    Create a single-scale OME-Zarr structure with dataset '0' and basic multiscales metadata.
    This avoids holding the full (C,Z,Y,X) array in RAM.
    """
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(out_zarr), mode="w")

    chunks = chunks_czyx or (1, 1, 256, 256)
    arr0 = root.create_dataset(
        "0",
        shape=shape_czyx,
        chunks=chunks,
        dtype=dtype,
        overwrite=True,
    )

    axes = in_meta.get("axes") or "czyx"
    voxel = in_meta.get("voxel_size_um") or {"x": 1.0, "y": 1.0, "z": 1.0}
    # scale vector must align with axes order
    scale_map = {"c": 1.0, "z": float(voxel.get("z") or 1.0), "y": float(voxel.get("y") or 1.0), "x": float(voxel.get("x") or 1.0)}
    scale_vec = [scale_map.get(a, 1.0) for a in axes]

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "image",
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": scale_vec}],
        }],
        "axes": [{"name": a, "type": ("channel" if a == "c" else "space")} for a in axes],
    }]

    # copy useful attrs
    root.attrs["channel_names"] = in_meta.get("channel_names")
    if in_meta.get("source_path"):
        root.attrs["source_path"] = in_meta.get("source_path")
    if in_meta.get("pft_meta"):
        root.attrs["pft_meta"] = in_meta.get("pft_meta")

    return arr0


def deconvolve_omezarr_3ch_to_omezarr(
    *,
    in_omezarr: Path,
    out_root: Path,
    model: PSFModel = "BW",
    iters: int = 15,
    background: float = 0.0,
    channel_wavelength_nm: dict[str, float] | None = None,
    overwrite: bool = True,
) -> DeconvRunInfo:
    """
    Core logic:
      - read OME-Zarr (expects CZYX)
      - for each channel: run DL2 RL with PSF(model, channel, lambda)
      - write result as OME-Zarr (single-scale) under out_root
    """
    in_omezarr = Path(in_omezarr)
    out_root = Path(out_root)
    channel_wavelength_nm = channel_wavelength_nm or dict(DEFAULT_CHANNEL_WAVELENGTH_NM)

    project_root = find_project_root(Path(__file__).resolve())

    if not in_omezarr.exists():
        raise FileNotFoundError(in_omezarr)

    # Read stable metadata
    meta = extract_ome_zarr_meta_for_compare(in_omezarr)  # includes channel_names, voxel_size_um, chunks, axes :contentReference[oaicite:3]{index=3}
    channel_names = meta.get("channel_names") or []
    if not isinstance(channel_names, list) or len(channel_names) < 3:
        raise ValueError(f"Expected >=3 channel_names in OME-Zarr attrs. Got: {channel_names}")

    # Open input data (prefer Zarr direct for chunked reads)
    root_in = zarr.open_group(str(in_omezarr), mode="r")
    arr_in = root_in["0"]  # expected C,Z,Y,X
    if arr_in.ndim != 4:
        raise ValueError(f"Expected input dataset '0' to be 4D (C,Z,Y,X). Got shape={arr_in.shape}")

    c, z, y, x = arr_in.shape

    # Build output folder & zarr
    tag = Path(in_omezarr).parent.name  # folder containing image.ome.zarr
    out_dir = out_root / f"{tag}__DL2_RL__PSF{model}__iter{iters}"
    out_zarr = out_dir / "image.ome.zarr"

    if overwrite:
        _ensure_empty_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_zarr.exists():
            raise FileExistsError(out_zarr)

    # Create output zarr dataset with same chunking where possible
    chunks_in = getattr(arr_in, "chunks", None)
    chunks_czyx = tuple(chunks_in) if chunks_in and len(chunks_in) == 4 else None

    arr_out = _write_single_scale_omezarr_like_input(
        out_zarr=out_zarr,
        in_meta=meta,
        shape_czyx=(c, z, y, x),
        dtype=np.dtype("uint16"),  # keep uint16 output unless you want float
        chunks_czyx=chunks_czyx,
    )

    # Setup Fiji + DL2 once
    fiji_dir = ensure_fiji_in_project(project_root, quiet=True)
    java = _find_java_exe(fiji_dir)
    dl2_jar = find_dl2_jar(fiji_dir)

    # Per-channel deconvolution
    for ch_idx in range(c):
        ch_name = channel_names[ch_idx]
        if ch_name not in channel_wavelength_nm:
            raise KeyError(f"No wavelength mapping for channel '{ch_name}'. Provide channel_wavelength_nm mapping.")

        lam = float(channel_wavelength_nm[ch_name])
        psf_path = _psf_path_for(model, ch_name, lam, project_root)
        if not psf_path.exists():
            raise FileNotFoundError(
                f"PSF not found: {psf_path}\n"
                f"Generate it first (PSFGenerator) for model={model}, channel={ch_name}, lambda={lam}."
            )

        # Read one channel volume (Z,Y,X) without loading all channels
        vol_zyx = np.asarray(arr_in[ch_idx, :, :, :])  # uint16

        # Write DL2 input TIFF for this channel (ImageJ-compatible; preserve uint16)
        inputs_dir = out_dir / "_inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        img_tif = inputs_dir / f"{tag}__C{ch_idx}__{ch_name}__image.tif"
        write_imagej_tiff_stack(vol_zyx, img_tif, axes="ZYX")  # from dl2.py :contentReference[oaicite:4]{index=4}

        # Run DL2 RL
        ch_out_dir = out_dir / f"_ch{ch_idx}_{ch_name}"
        run_dl2_cli(
            java=java,
            dl2_jar=dl2_jar,
            image_tif=img_tif,
            psf_tif=psf_path,
            out_dir=ch_out_dir,
            iterations=iters,
            background=background,
        )  

        # Collect output stack from DL2 folder:
        # DL2 output naming can vary; we pick first TIFF that looks like a stack (not PSF/input)
        outs = sorted(list(ch_out_dir.glob("*.tif")) + list(ch_out_dir.glob("*.tiff")))
        if not outs:
            raise FileNotFoundError(f"No output TIFFs written by DL2 in {ch_out_dir}")

        out_tif = max(outs, key=lambda p: p.stat().st_size)
        deconv_zyx = tiff.imread(str(out_tif))

        if deconv_zyx.shape != (z, y, x):
            raise ValueError(f"DL2 output shape mismatch for channel {ch_idx}: {deconv_zyx.shape} vs {(z,y,x)}")

        # Write into output zarr channel
        arr_out[ch_idx, :, :, :] = deconv_zyx.astype(np.uint16, copy=False)

    return DeconvRunInfo(
        in_zarr=in_omezarr,
        out_zarr=out_zarr,
        out_dir=out_dir,
        model=model,
        iters=iters,
        background=background,
    )