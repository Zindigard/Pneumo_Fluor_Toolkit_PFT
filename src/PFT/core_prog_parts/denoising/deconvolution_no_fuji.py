from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile as tiff
import zarr
from skimage.restoration import richardson_lucy

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.decoder_omezar import (
    extract_ome_zarr_meta_for_compare,
)

PSFModel = Literal["BW", "GL", "RW"]

"""
Richardson-Lucy deconvolution on OME-Zarr using scikit-image.
"""

DEFAULT_CHANNEL_WAVELENGTH_NM = {
    "TV1-T1-SR": 405.0,
    "TV1-T2-SR": 488.0,
    "TV1-T3-SR": 561.0,
}


@dataclass(frozen=True)
class SkimageDeconvRunInfo:
    in_zarr: Path
    out_zarr: Path
    out_dir: Path
    model: PSFModel
    iters: int
    background: float
    level: int
    out_dtype: str
    clip: bool
    filter_epsilon: float | None


def _dataset_path_for_level(zarr_dir: Path, level: int) -> str:
    """Return NGFF dataset path for a given pyramid level."""
    root = zarr.open_group(str(zarr_dir), mode="r")
    ms = root.attrs.get("multiscales")

    if not ms or not isinstance(ms, list) or not ms[0].get("datasets"):
        if level != 0:
            raise ValueError("No multiscales found; only level=0 is available for this OME-Zarr.")
        return "0"

    ds = ms[0]["datasets"]
    levels = list(range(len(ds)))
    if level not in levels:
        raise ValueError(f"Invalid level={level}. Available levels: {levels}")

    path = ds[level].get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Invalid dataset path for level={level}: {path!r}")
    return path


def _psf_path_for(
    *,
    model: PSFModel,
    channel_name: str,
    wavelength_nm: float,
    level: int,
    project_root: Path,
) -> Path:
    """
    Expected PSF naming convention:
      psf_<MODEL>_<CHANNEL>_Lambda<NNN>nm__L<level>.tif
    """
    psf_dir = project_root / "results" / "psf" / "generated"
    return psf_dir / f"psf_{model}_{channel_name}_Lambda{int(round(wavelength_nm))}nm__L{int(level)}.tif"


def _ensure_empty_dir(p: Path) -> None:
    """Remove and recreate a directory."""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    """
    Make PSF valid for Richardson-Lucy:
    - float32
    - finite
    - non-negative
    - sum == 1
    """
    psf = np.asarray(psf, dtype=np.float32)

    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D (Z,Y,X). Got shape={psf.shape}")

    if not np.isfinite(psf).all():
        raise ValueError("PSF contains NaN or Inf values.")

    psf = np.maximum(psf, 0.0)
    s = float(psf.sum())
    if s <= 0:
        raise ValueError("PSF sum <= 0 after clipping negatives.")

    return psf / s


def _prepare_image_for_rl(image_zyx: np.ndarray, background: float) -> np.ndarray:
    """
    Prepare image for Richardson-Lucy:
    - convert to float32
    - subtract constant background
    - clamp to non-negative
    """
    img = np.asarray(image_zyx, dtype=np.float32)

    if img.ndim != 3:
        raise ValueError(f"Image must be 3D (Z,Y,X). Got shape={img.shape}")

    if not np.isfinite(img).all():
        raise ValueError("Image contains NaN or Inf values.")

    img = img - np.float32(background)
    img = np.maximum(img, 0.0)
    return img


def _create_output_omezarr_single_scale_like_input(
    *,
    out_zarr: Path,
    in_meta: dict,
    shape_czyx: tuple[int, int, int, int],
    dtype: np.dtype,
    chunks_czyx: tuple[int, int, int, int] | None,
) -> zarr.Array:
    """
    Create output OME-Zarr as single-scale dataset '0',
    preserving basic metadata from the input.
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

    axes = str(in_meta.get("axes") or "czyx").lower()
    voxel = in_meta.get("voxel_size_um") or {"x": 1.0, "y": 1.0, "z": 1.0}

    scale_map = {
        "c": 1.0,
        "z": float(voxel.get("z") or 1.0),
        "y": float(voxel.get("y") or 1.0),
        "x": float(voxel.get("x") or 1.0),
    }
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

    if in_meta.get("channel_names") is not None:
        root.attrs["channel_names"] = in_meta["channel_names"]

    if in_meta.get("source_path") is not None:
        root.attrs["source_path"] = in_meta["source_path"]

    root.attrs["pft_deconvolution_method"] = "skimage_richardson_lucy"

    return arr0


def _print_omezarr_meta(title: str, zarr_dir: Path, *, level: int) -> None:
    """Print compact OME-Zarr metadata for debugging."""
    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    print(f"\n[{title}] OME-Zarr meta")
    print(f"  path         : {zarr_dir}")
    print(f"  level        : {level}")
    print(f"  array_path   : {meta.get('array_path')}")
    print(f"  stored shape : {meta.get('shape')}")
    print(f"  stored dtype : {meta.get('dtype')}")
    print(f"  stored chunks: {meta.get('chunks')}")
    print(f"  axes         : {meta.get('axes')}")
    print(f"  voxel_size_um: {meta.get('voxel_size_um')}")
    print(f"  channel_names: {meta.get('channel_names')}")


def deconvolve_omezarr_3ch_to_omezarr_skimage(
    *,
    in_omezarr: Path,
    out_root: Path,
    model: PSFModel = "BW",
    iters: int = 15,
    background: float = 0.0,
    level: int = 0,
    channel_wavelength_nm: dict[str, float] | None = None,
    overwrite: bool = True,
    clip: bool = False,
    filter_epsilon: float | None = None,
) -> SkimageDeconvRunInfo:
    """
    Run scikit-image Richardson-Lucy deconvolution on each channel of a 3D OME-Zarr.

    Parameters
    ----------
    in_omezarr : Path
        Input OME-Zarr path.
    out_root : Path
        Root directory where the deconvolved output folder is created.
    model : {"BW","GL","RW"}
        PSF model name.
    iters : int
        Number of RL iterations.
    background : float
        Constant background subtracted before deconvolution.
    level : int
        OME-Zarr pyramid level to read from input.
    channel_wavelength_nm : dict[str, float] | None
        Mapping from channel name to wavelength.
    overwrite : bool
        Whether to overwrite output directory.
    clip : bool
        Passed to skimage.restoration.richardson_lucy.
    filter_epsilon : float | None
        Small stability parameter passed to skimage RL.

    Returns
    -------
    SkimageDeconvRunInfo
        Metadata about the completed run.
    """
    in_omezarr = Path(in_omezarr)
    out_root = Path(out_root)
    channel_wavelength_nm = channel_wavelength_nm or dict(DEFAULT_CHANNEL_WAVELENGTH_NM)

    if iters < 1:
        raise ValueError(f"`iters` must be >= 1. Got {iters}")

    project_root = find_project_root(Path(__file__).resolve())

    _print_omezarr_meta("INPUT", in_omezarr, level=level)

    dataset_path = _dataset_path_for_level(in_omezarr, level)
    root_in = zarr.open_group(str(in_omezarr), mode="r")
    arr_in = root_in[dataset_path]

    print(f"\n[INPUT] raw dataset")
    print(f"  dataset_path: {dataset_path}")
    print(f"  shape       : {arr_in.shape}")
    print(f"  dtype       : {arr_in.dtype}")

    if arr_in.ndim != 4:
        raise ValueError(
            f"Expected dataset '{dataset_path}' to be 4D (C,Z,Y,X) at level={level}. Got shape={arr_in.shape}"
        )

    c, z, y, x = arr_in.shape

    meta = extract_ome_zarr_meta_for_compare(in_omezarr, level=level)
    channel_names = meta.get("channel_names") or []
    if not isinstance(channel_names, list) or len(channel_names) < c:
        raise ValueError(f"Expected channel_names list length >= C. Got: {channel_names}")

    tag = in_omezarr.parent.name
    out_dir = out_root / f"{tag}__SK_RL__PSF{model}__iter{iters}__L{int(level)}"
    out_zarr = out_dir / "image.ome.zarr"

    if overwrite:
        _ensure_empty_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_zarr.exists():
            raise FileExistsError(out_zarr)

    chunks_in = getattr(arr_in, "chunks", None)
    chunks_czyx = tuple(chunks_in) if chunks_in and len(chunks_in) == 4 else None

    arr_out = _create_output_omezarr_single_scale_like_input(
        out_zarr=out_zarr,
        in_meta=meta,
        shape_czyx=(c, z, y, x),
        dtype=np.dtype(np.float32),
        chunks_czyx=chunks_czyx,
    )

    print(f"\n[OUTPUT] created output OME-Zarr (float32)")
    print(f"  out_dir : {out_dir}")
    print(f"  out_zarr: {out_zarr}")
    print(f"  out dataset '0' shape: {arr_out.shape} dtype: {arr_out.dtype}")

    for ch_idx in range(c):
        ch_name = str(channel_names[ch_idx])
        if ch_name not in channel_wavelength_nm:
            raise KeyError(
                f"No wavelength mapping for channel '{ch_name}'. Provide channel_wavelength_nm mapping.\n"
                f"Known keys: {sorted(channel_wavelength_nm.keys())}"
            )

        lam = float(channel_wavelength_nm[ch_name])
        psf_path = _psf_path_for(
            model=model,
            channel_name=ch_name,
            wavelength_nm=lam,
            level=level,
            project_root=project_root,
        )
        if not psf_path.exists():
            raise FileNotFoundError(
                f"PSF not found: {psf_path}\n"
                f"Generate it first for model={model}, channel={ch_name}, lambda={lam}, level={level}."
            )

        vol_zyx = np.asarray(arr_in[ch_idx, :, :, :])
        vol_zyx_f = _prepare_image_for_rl(vol_zyx, background=background)

        psf_zyx = tiff.imread(str(psf_path))
        psf_zyx_f = _normalize_psf(psf_zyx)

        print(f"\n[CHANNEL {ch_idx}] {ch_name}")
        print(
            f"  input vol shape: {vol_zyx.shape} dtype={vol_zyx.dtype} "
            f"min={float(vol_zyx.min()):.3g} max={float(vol_zyx.max()):.3g}"
        )
        print(
            f"  prepared input dtype={vol_zyx_f.dtype} "
            f"min={float(vol_zyx_f.min()):.3g} max={float(vol_zyx_f.max()):.3g}"
        )
        print(
            f"  PSF path: {psf_path}\n"
            f"  PSF shape: {psf_zyx_f.shape} dtype={psf_zyx_f.dtype} "
            f"sum={float(psf_zyx_f.sum()):.6f}"
        )

        if psf_zyx_f.shape != vol_zyx_f.shape and any(p > i for p, i in zip(psf_zyx_f.shape, vol_zyx_f.shape)):
            raise ValueError(
                f"PSF shape {psf_zyx_f.shape} is larger than image shape {vol_zyx_f.shape} "
                f"along at least one axis."
            )

        deconv_zyx = richardson_lucy(
            image=vol_zyx_f,
            psf=psf_zyx_f,
            num_iter=int(iters),
            clip=bool(clip),
            filter_epsilon=filter_epsilon,
        )

        deconv_zyx = np.asarray(deconv_zyx, dtype=np.float32)

        print(
            f"  RL output: shape={deconv_zyx.shape} dtype={deconv_zyx.dtype} "
            f"min={float(deconv_zyx.min()):.3g} max={float(deconv_zyx.max()):.3g}"
        )

        if deconv_zyx.shape != (z, y, x):
            raise ValueError(
                f"RL output shape mismatch for channel {ch_idx}: {deconv_zyx.shape} vs {(z, y, x)}"
            )

        arr_out[ch_idx, :, :, :] = np.ascontiguousarray(deconv_zyx)

        check = np.asarray(arr_out[ch_idx, :, :, :])
        print(
            f"  [WRITE CHECK] stored dtype={check.dtype} "
            f"min={float(check.min()):.3g} max={float(check.max()):.3g}"
        )

    _print_omezarr_meta("OUTPUT-STORED", out_zarr, level=0)

    return SkimageDeconvRunInfo(
        in_zarr=in_omezarr,
        out_zarr=out_zarr,
        out_dir=out_dir,
        model=model,
        iters=iters,
        background=background,
        level=int(level),
        out_dtype="float32",
        clip=bool(clip),
        filter_epsilon=filter_epsilon,
    )


__all__ = [
    "PSFModel",
    "DEFAULT_CHANNEL_WAVELENGTH_NM",
    "SkimageDeconvRunInfo",
    "deconvolve_omezarr_3ch_to_omezarr_skimage",
]