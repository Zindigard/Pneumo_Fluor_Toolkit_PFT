from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.fuji_managment import find_project_root, ensure_fiji_in_project
from PFT.core_prog_parts.psf_creator import _find_java_exe


# -------------------------
# Small memory helper (optional)
# -------------------------
def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _fmt_mb(x: Optional[int]) -> str:
    return "n/a" if x is None else f"{x / (1024**2):.1f} MB"


# -------------------------
# DL2 discovery
# -------------------------
def find_dl2_jar(fiji_dir: Path) -> Path:
    plugins = fiji_dir / "plugins"
    candidates = [
        plugins / "DeconvolutionLab_2.jar",
        plugins / "DeconvolutionLab2.jar",
        plugins / "DeconvolutionLab.jar",
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = list(plugins.glob("**/*Deconvolution*Lab*.jar")) + list(plugins.glob("**/*DeconvolutionLab*.jar"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"DeconvolutionLab2 jar not found under: {plugins}")


# -------------------------
# OME-Zarr loading (channel -> ZYX)
# -------------------------
def load_omezarr_channel_zyx(image_omezarr_dir: Path, channel_index: int = 0) -> np.ndarray:
    root = zarr.open(str(image_omezarr_dir), mode="r")
    if "0" not in root:
        raise KeyError(f"Zarr array '0' not found inside: {image_omezarr_dir}")

    arr = root["0"]  # expected (C,Z,Y,X)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (C,Z,Y,X). Got shape={arr.shape}")

    if not (0 <= channel_index < arr.shape[0]):
        raise IndexError(f"Channel index {channel_index} out of range for shape {arr.shape}")

    vol = np.asarray(arr[channel_index, :, :, :])  # (Z,Y,X)
    return vol


# -------------------------
# Validation (no changes)
# -------------------------
def basic_validity_checks(image_zyx: np.ndarray, psf_path: Path) -> None:
    if image_zyx.ndim != 3:
        raise ValueError(f"Image must be 3D (Z,Y,X). Got shape={image_zyx.shape}")
    if not psf_path.exists():
        raise FileNotFoundError(psf_path)

    if not np.isfinite(image_zyx.astype(np.float64, copy=False)).all():
        raise ValueError("Image contains NaN/Inf.")
    if float(np.max(image_zyx)) <= 0:
        raise ValueError("Image max <= 0 (looks empty).")


# -------------------------
# TIFF writing (ImageJ stack; PRESERVE uint16)
# -------------------------
def write_imagej_tiff_stack(arr: np.ndarray, out_path: Path, axes: str = "ZYX") -> None:
    """
    Write an ImageJ-compatible TIFF stack so DeconvolutionLab2 can open it.
    Preserves dtype and values (no scaling, no normalization, no float conversion).
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


# -------------------------
# Run DL2 (Richardson–Lucy ONLY) using ImageJ-style "file <path>"
# -------------------------
def run_dl2_cli(
    java: Path,
    dl2_jar: Path,
    image_tif: Path,
    psf_tif: Path,
    out_dir: Path,
    iterations: int = 1,
    background: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT (Windows/DL2): match ImageJ working syntax exactly:
    # -image file <path>   -psf file <path>
    cmd = [
        str(java),
        "-jar",
        str(dl2_jar),
        "Run",
        "-image",
        "file",
        str(image_tif),
        "-psf",
        "file",
        str(psf_tif),
        # Hard-lock Richardson–Lucy
        "-algorithm",
        "RL",
        str(background),
        str(iterations),
        "-out",
        "mip",
        "MLI",
        "-path",
        str(out_dir),
    ]

    print("Running DL2:\n ", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n--- DL2 STDOUT ---\n", res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"DL2 failed.\n--- STDERR ---\n{res.stderr}")


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--omezarr_dir",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\3d_data\20220218_dynamic\DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM\image.ome.zarr",
    )
    ap.add_argument("--channel", type=int, default=0, help="Channel index.")
    ap.add_argument(
        "--psf_tif",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\generated\psf_BW_TV1-T1-SR_Lambda405nm.tif",
    )
    ap.add_argument(
        "--out_dir",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\deconv",
        help="Root output directory (outputs will be stored here).",
    )
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--background", type=float, default=0.0)
    ap.add_argument(
        "--tag",
        default="DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM",
        help="Name used to build folder/file names.",
    )

    args = ap.parse_args()

    # memory baseline (optional)
    t0 = time.time()
    psutil = _try_import_psutil()
    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else None

    # setup Fiji/Java/DL2
    start_path = Path(__file__)
    project_root = find_project_root(start_path.resolve())
    fiji_dir = ensure_fiji_in_project(project_root, quiet=True)
    java = _find_java_exe(fiji_dir)
    dl2_jar = find_dl2_jar(fiji_dir)

    omezarr_dir = Path(args.omezarr_dir)
    psf_path = Path(args.psf_tif)
    out_root = Path(args.out_dir)

    if not omezarr_dir.exists():
        raise FileNotFoundError(omezarr_dir)
    if not psf_path.exists():
        raise FileNotFoundError(psf_path)

    # load image (NO changes)
    image_zyx = load_omezarr_channel_zyx(omezarr_dir, channel_index=args.channel)
    basic_validity_checks(image_zyx, psf_path)

    # PSF info (read-only, does not resave)
    psf_arr = tiff.imread(str(psf_path))

    print("Input summary (no changes applied):")
    print(f"  Image: shape={image_zyx.shape}, dtype={image_zyx.dtype}, min={image_zyx.min()}, max={image_zyx.max()}")
    print(f"  PSF  : path={psf_path}")
    print(f"        shape={psf_arr.shape}, dtype={psf_arr.dtype}, min={psf_arr.min()}, max={psf_arr.max()}, sum={float(np.sum(psf_arr))}")

    # NO TEMP:
    #   - save ONLY the image stack into run_dir/_inputs (as uint16, preserved)
    #   - use PSF directly from generated/ path
    run_dir = out_root / f"{args.tag}__C{args.channel}__iter{args.iters}"
    inputs_dir = run_dir / "_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    img_tif = inputs_dir / f"{args.tag}__C{args.channel}__image.tif"

    # Preserve uint16 (no float conversion)
    write_imagej_tiff_stack(image_zyx, img_tif, axes="ZYX")

    # Run DL2 RL; outputs saved in run_dir
    run_dl2_cli(
        java=java,
        dl2_jar=dl2_jar,
        image_tif=img_tif,
        psf_tif=psf_path,  # DO NOT resave PSF
        out_dir=run_dir,
        iterations=args.iters,
        background=args.background,
    )

    print("\nDL2 finished.")
    print(f"  Run folder (outputs): {run_dir}")
    print(f"  Image input saved in : {img_tif}")
    print(f"  PSF used from        : {psf_path}")

    # memory end (optional)
    rss1 = proc.memory_info().rss if proc else None
    print(f"\nMemory RSS: start={_fmt_mb(rss0)} end={_fmt_mb(rss1)}  elapsed={time.time() - t0:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())