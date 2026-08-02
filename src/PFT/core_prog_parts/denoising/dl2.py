"""Provide command-line and programmatic utilities for dl2."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.denoising.fuji_managment import (
    find_project_root,
    ensure_fiji_in_project,
    ensure_deconvolutionlab2_exists,
)
from PFT.core_prog_parts.denoising.psf_creator import _find_java_exe
from PFT.core_prog_parts.decoder_omezar import (
    load_ome_zarr_3d_czyx,
    extract_ome_zarr_meta_for_compare,
)

"Logic of running DeconvolutionLab2 RL deconvolution on OME-Zarr data, with checks and reporting"

def _try_import_psutil():
    """Internal helper used by this module.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = _try_import_psutil()
    """
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def mem_box(
    title: str,
    t0: float,
    rss0: Optional[int],
    rss1: Optional[int],
    peak_rss: Optional[int],
    py_peak_bytes: Optional[int],
) -> str:
    """Helper function used by this module.

    Args:
        title (str): Title displayed on the generated figure or report section.
        t0 (float): Numerical value controlling t0.
        rss0 (Optional[int]): Numerical value controlling rss0.
        rss1 (Optional[int]): Numerical value controlling rss1.
        peak_rss (Optional[int]): Numerical value controlling peak rss.
        py_peak_bytes (Optional[int]): Numerical value controlling py peak bytes.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = mem_box(
        ...     title="title",
        ...     t0=0.5,
        ...     rss0=1,
        ...     rss1=1,
        ...     peak_rss=1,
        ...     py_peak_bytes=1,
        ... )
    """
    def fmt_mb(x: Optional[int]) -> str:
        """Helper function used by this module.

        Args:
            x (Optional[int]): Horizontal coordinate or numerical input value used by the operation.

        Returns:
            str: Generated or resolved text value.

        Example:
            >>> result = fmt_mb(x=1)
        """
        return "n/a" if x is None else f"{x / (1024**2):.1f} MB"

    dt = time.time() - t0
    lines = [
        "┌" + "─" * 46 + "┐",
        f"│ {title:<44} │",
        "├" + "─" * 46 + "┤",
        f"│ elapsed: {dt:>7.2f} s{'':<31}│",
        f"│ RSS start: {fmt_mb(rss0):>10}   RSS end: {fmt_mb(rss1):>10} │",
        f"│ Peak RSS:  {fmt_mb(peak_rss):>10}   Py peak: {fmt_mb(py_peak_bytes):>10} │",
        "└" + "─" * 46 + "┘",
    ]
    return "\n".join(lines)


def _infer_lambda_nm_from_channel_name(name: Optional[str]) -> Optional[int]:
    """T1 -> 405 nm, T2 -> 488 nm, T3 -> 561 nm.

    Args:
        name (Optional[str]): Name used to identify the current object, resource, or output.

    Returns:
        Optional[int]: Computed numerical result.

    Example:
        >>> result = _infer_lambda_nm_from_channel_name(name="name")
    """
    if not name:
        return None
    n = name.upper()
    if "T1" in n:
        return 405
    if "T2" in n:
        return 488
    if "T3" in n:
        return 561
    return None


PSFMode = Literal["file", "auto"]
PSFModel = Literal["BW", "GL", "RW"]


def _select_psf_path(
    *,
    psf_mode: PSFMode,
    psf_tif: Optional[Path],
    psf_dir: Optional[Path],
    psf_model: Optional[str],
    channel_name: Optional[str],
    lambda_nm: Optional[int],
    level: Optional[int] = None,
) -> Path:
    
    """Internal helper used by this module.

    Args:
        psf_mode (PSFMode): Value specifying point-spread function mode for the operation.
        psf_tif (Optional[Path]): Filesystem path used for point-spread function TIFF data.
        psf_dir (Optional[Path]): Directory used for point-spread function.
        psf_model (Optional[str]): Text value specifying point-spread function model.
        channel_name (Optional[str]): Text value specifying channel name.
        lambda_nm (Optional[int]): Numerical value controlling lambda nm.
        level (Optional[int]): Numerical value controlling level. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _select_psf_path(
        ...     psf_mode=...,
        ...     psf_tif=Path("path/to/resource"),
        ...     psf_dir=Path("path/to/resource"),
        ...     psf_model="psf_model",
        ...     channel_name="channel_name",
        ...     lambda_nm=1,
        ... )
    """
    if psf_mode == "file":
        if psf_tif is None:
            raise ValueError("--psf_mode file requires --psf_tif")
        return psf_tif

    if psf_dir is None:
        raise ValueError("--psf_mode auto requires --psf_dir")
    if psf_model is None:
        raise ValueError("--psf_mode auto requires --psf_model (BW/GL/RW)")

    psf_dir = Path(psf_dir)
    if not psf_dir.exists():
        raise FileNotFoundError(f"PSF dir not found: {psf_dir}")

    nm = int(lambda_nm) if lambda_nm is not None else _infer_lambda_nm_from_channel_name(channel_name)
    if nm is None:
        raise ValueError(
            "Could not infer wavelength for auto PSF selection. "
            "Provide --lambda_nm, or ensure the channel name contains T1/T2/T3."
        )

    model = psf_model.upper().strip()

    patterns: list[str] = []
    if level is not None:
        patterns.append(f"psf_{model}_*Lambda{nm}nm__L{int(level)}.tif")
        patterns.append(f"*{model}*Lambda{nm}nm*__L{int(level)}*.tif")
    patterns.append(f"psf_{model}_*Lambda{nm}nm.tif")
    patterns.append(f"*{model}*Lambda{nm}nm*.tif")

    for patt in patterns:
        matches = sorted(psf_dir.glob(patt))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No PSF found for model={model} lambda={nm}nm (level={level}) in {psf_dir}\n"
        f"Tried patterns:\n  - " + "\n  - ".join(patterns)
    )


def load_omezarr_channel_zyx(
    image_omezarr_dir: Path,
    *,
    channel_index: int = 0,
    level: int = 0,
    time: Optional[int] = 0,
) -> tuple[np.ndarray, Optional[str]]:
   
    """Load data and return the processed result.

    Args:
        image_omezarr_dir (Path): Directory used for image OME-Zarr.
        channel_index (int): Zero-based index selecting channel. Defaults to ``0``.
        level (int): Numerical value controlling level. Defaults to ``0``.
        time (Optional[int]): Numerical value controlling time. Defaults to ``0``.

    Returns:
        tuple[np.ndarray, Optional[str]]: Collection containing the generated or selected values.

    Raises:
        IndexError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = load_omezarr_channel_zyx(image_omezarr_dir=Path("path/to/resource"))
    """
    image_omezarr_dir = Path(image_omezarr_dir)

    meta = extract_ome_zarr_meta_for_compare(image_omezarr_dir, level=level)
    ch_names = meta.get("channel_names")
    channel_name = None
    if isinstance(ch_names, list) and 0 <= channel_index < len(ch_names):
        channel_name = str(ch_names[channel_index])

    vol_czyx, _axes = load_ome_zarr_3d_czyx(
        image_omezarr_dir,
        level=level,
        time=time,
        channels=None,
        as_numpy=True,
    )
    if not (0 <= channel_index < vol_czyx.shape[0]):
        raise IndexError(f"Channel index {channel_index} out of range for vol shape {vol_czyx.shape}")

    return np.asarray(vol_czyx[channel_index]), channel_name


def basic_validity_checks(image_zyx: np.ndarray, psf: np.ndarray) -> None:
    """Helper function used by this module.

    Args:
        image_zyx (np.ndarray): Array containing image zyx.
        psf (np.ndarray): Array containing point-spread function.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> basic_validity_checks(image_zyx=image_array, psf=image_array)
    """
    if image_zyx.ndim != 3:
        raise ValueError(f"Image must be 3D (Z,Y,X). Got shape={image_zyx.shape}")
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D (Z,Y,X). Got shape={psf.shape}")

    if not np.isfinite(image_zyx.astype(np.float64, copy=False)).all():
        raise ValueError("Image contains NaN/Inf.")
    if not np.isfinite(psf.astype(np.float64, copy=False)).all():
        raise ValueError("PSF contains NaN/Inf.")

    if float(np.max(image_zyx)) <= 0:
        raise ValueError("Image max <= 0 (looks empty).")
    if float(np.max(psf)) <= 0:
        raise ValueError("PSF max <= 0 (looks empty).")

    if float(np.min(psf)) < 0:
        print("WARNING: PSF has negative values. RL usually expects PSF >= 0.")


def write_imagej_tiff_stack(arr: np.ndarray, out_path: Path, axes: str = "ZYX") -> None:
   
    """Write the requested report or metadata file.

    Args:
        arr (np.ndarray): Array containing arr.
        out_path (Path): Filesystem path associated with out.
        axes (str): Axis specification describing the dimensional order of the image data. Defaults to ``"ZYX"``.

    Example:
        >>> write_imagej_tiff_stack(arr=image_array, out_path=Path("path/to/resource"))
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr)
    tiff.imwrite(
        str(out_path),
        arr,
        imagej=True,
        metadata={"axes": axes},
        photometric="minisblack",
    )


def run_dl2_cli(
    *,
    java: Path,
    dl2_jar: Path,
    image_tif: Path,
    psf_tif: Path,
    out_dir: Path,
    iterations: int = 1,
    background: float = 0.0,
) -> Path:
    
    """Run the main processing step for this workflow.

    Args:
        java (Path): Filesystem path used for java.
        dl2_jar (Path): Filesystem path used for dl2 jar.
        image_tif (Path): Filesystem path used for image TIFF data.
        psf_tif (Path): Filesystem path used for point-spread function TIFF data.
        out_dir (Path): Directory used for out.
        iterations (int): Numerical value controlling iterations. Defaults to ``1``.
        background (float): Numerical value controlling background. Defaults to ``0.0``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = run_dl2_cli(
        ...     java=Path("path/to/resource"),
        ...     dl2_jar=Path("path/to/resource"),
        ...     image_tif=Path("path/to/resource"),
        ...     psf_tif=Path("path/to/resource"),
        ...     out_dir=Path("path/to/resource"),
        ... )
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    img = tiff.imread(str(image_tif))
    psf = tiff.imread(str(psf_tif))

    print("\n[DL2 PRE-CHECK]")
    print(f"  Image dtype: {img.dtype}, shape={img.shape}, min={img.min()}, max={img.max()}")
    print(f"  PSF   dtype: {psf.dtype}, shape={psf.shape}, min={psf.min()}, max={psf.max()}, sum={float(np.sum(psf))}")

    if not (np.issubdtype(img.dtype, np.integer) or np.issubdtype(img.dtype, np.floating)):
        raise ValueError(f"Unsupported image dtype: {img.dtype}")
    if not (np.issubdtype(psf.dtype, np.integer) or np.issubdtype(psf.dtype, np.floating)):
        raise ValueError(f"Unsupported PSF dtype: {psf.dtype}")

    cmd = [
        str(java),
        "-jar",
        str(dl2_jar),
        "Run",
        "-image", "file", str(image_tif),
        "-psf", "file", str(psf_tif),
        "-algorithm", "RL", str(iterations), str(background),
        "-out", "stack", "DL2", "intact",
        "-path", str(out_dir),
    ]

    print("\nRunning DL2:\n ", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("\n--- DL2 STDOUT ---\n", res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"DL2 failed.\n--- STDERR ---\n{res.stderr}")

    outs = sorted(list(out_dir.glob("*.tif")) + list(out_dir.glob("*.tiff")))
    if not outs:
        raise FileNotFoundError(f"No TIFF output written by DL2 in {out_dir}")

    out_tif = max(outs, key=lambda p: p.stat().st_size)

    out_img = tiff.imread(str(out_tif))

    print("\n[DL2 POST-CHECK]")
    print(f"  Output file : {out_tif.name}")
    print(f"  Output dtype: {out_img.dtype}")
    print(f"  Output shape: {out_img.shape}")
    print(f"  Output min/max: {out_img.min()} / {out_img.max()}")

    if out_img.ndim != 3:
        raise ValueError(f"DL2 output is not 3D (Z,Y,X). Got shape={out_img.shape}")

    if out_img.shape != img.shape:
        print(f"WARNING: Output shape differs from input! in={img.shape} out={out_img.shape}")

    if img.dtype != out_img.dtype:
        print(f"NOTE: DL2 changed dtype: {img.dtype} -> {out_img.dtype}")

    return out_tif


def main() -> int:
    """Helper function used by this module.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    ap = argparse.ArgumentParser(description="Run DeconvolutionLab2 Richardson–Lucy on ONE channel of an OME-Zarr.")
    ap.add_argument(
        "--omezarr_dir",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\3d_data\20220218_dynamic\DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM\image.ome.zarr",
    )
    ap.add_argument("--channel", type=int, default=0)

    ap.add_argument("--level", type=int, default=0, help="OME-Zarr pyramid level")
    ap.add_argument("--time", type=int, default=0, help="Time index if data has T axis (ignored if no T axis)")

    ap.add_argument("--psf_mode", choices=["file", "auto"], default="file")
    ap.add_argument("--psf_tif", default=None, help="Used when --psf_mode file")
    ap.add_argument("--psf_dir", default=None, help="Directory with generated PSFs (used when --psf_mode auto)")
    ap.add_argument("--psf_model", default="BW", help="BW/GL/RW (used when --psf_mode auto)")
    ap.add_argument(
        "--lambda_nm",
        type=int,
        default=None,
        help="Force wavelength (nm) for auto PSF selection. If not set, inferred from channel name (T1/T2/T3).",
    )

    ap.add_argument("--out_dir", default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\deconv")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--background", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    psutil = _try_import_psutil()
    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else None

    py_peak = None
    try:
        import tracemalloc

        tracemalloc.start()
    except Exception:
        tracemalloc = None  # type: ignore

    project_root = find_project_root(Path(__file__).resolve())
    fiji_dir = ensure_fiji_in_project(project_root, quiet=True)

    dl2_jar = ensure_deconvolutionlab2_exists(fiji_dir, auto_download=True, quiet=True)
    java = _find_java_exe(fiji_dir)

    omezarr_dir = Path(args.omezarr_dir)
    out_root = Path(args.out_dir)

    image_zyx, channel_name = load_omezarr_channel_zyx(
        omezarr_dir,
        channel_index=args.channel,
        level=args.level,
        time=args.time,
    )

    psf_path = _select_psf_path(
        psf_mode=args.psf_mode,
        psf_tif=Path(args.psf_tif) if args.psf_tif else None,
        psf_dir=Path(args.psf_dir) if args.psf_dir else None,
        psf_model=args.psf_model,
        channel_name=channel_name,
        lambda_nm=args.lambda_nm,
        level=args.level,
    )

    psf = tiff.imread(str(psf_path))
    basic_validity_checks(image_zyx, psf)

    print("\nInput summary (no changes applied):")
    print(f"  OME-Zarr        : {omezarr_dir}")
    print(f"  level/time      : {args.level}/{args.time}")
    print(f"  channel idx/name: {args.channel}/{channel_name}")
    print(f"  Image           : shape={image_zyx.shape}, dtype={image_zyx.dtype}, min={image_zyx.min()}, max={image_zyx.max()}")
    print(f"  PSF file        : {psf_path}")
    print(f"  PSF             : shape={psf.shape}, dtype={psf.dtype}, min={psf.min()}, max={psf.max()}, sum={float(np.sum(psf))}")

    with tempfile.TemporaryDirectory(prefix="pft_dl2_") as td:
        td = Path(td)
        tmp_img = td / f"tmp_image_C{args.channel}.tif"
        tmp_psf = td / "tmp_psf.tif"

        write_imagej_tiff_stack(image_zyx, tmp_img, axes="ZYX")
        write_imagej_tiff_stack(psf, tmp_psf, axes="ZYX")

        out_dir = out_root / f"{omezarr_dir.parent.name}__C{args.channel}__iter{args.iters}"
        out_tif = run_dl2_cli(
            java=java,
            dl2_jar=dl2_jar,
            image_tif=tmp_img,
            psf_tif=tmp_psf,
            out_dir=out_dir,
            iterations=args.iters,
            background=args.background,
        )

        print(f"\nDL2 finished.")
        print(f"  Output folder: {out_dir}")
        print(f"  Output file  : {out_tif}")

    rss1 = proc.memory_info().rss if proc else None
    peak_rss = None
    if proc and psutil:
        try:
            peak_rss = proc.memory_info().peak_wset
        except Exception:
            peak_rss = None

    if "tracemalloc" in globals() and tracemalloc:
        try:
            _, peak = tracemalloc.get_traced_memory()
            py_peak = int(peak)
            tracemalloc.stop()
        except Exception:
            py_peak = None

    print(mem_box("DL2 RL run", t0, rss0, rss1, peak_rss, py_peak))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())