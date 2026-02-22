from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff
import zarr

from PFT.core_prog_parts.fuji_managment import (
    find_project_root,
    ensure_fiji_in_project,
    ensure_deconvolutionlab2_exists,
)
from PFT.core_prog_parts.psf_creator import _find_java_exe


def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def mem_box(title: str, t0: float, rss0: Optional[int], rss1: Optional[int], peak_rss: Optional[int],
            py_peak_bytes: Optional[int]) -> str:
    def fmt_mb(x: Optional[int]) -> str:
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


def load_omezarr_channel_zyx(image_omezarr_dir: Path, channel_index: int = 0) -> np.ndarray:
    root = zarr.open(str(image_omezarr_dir), mode="r")
    if "0" not in root:
        raise KeyError(f"Zarr array '0' not found inside: {image_omezarr_dir}")

    arr = root["0"]  # expected (C,Z,Y,X)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (C,Z,Y,X). Got shape={arr.shape}")

    if not (0 <= channel_index < arr.shape[0]):
        raise IndexError(f"Channel index {channel_index} out of range for shape {arr.shape}")

    return np.asarray(arr[channel_index, :, :, :])  # (Z,Y,X)


def basic_validity_checks(image_zyx: np.ndarray, psf: np.ndarray) -> None:
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
    """
    ImageJ-compatible TIFF stack. Preserves dtype and values (no scaling).
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
) -> None:
    """
    DL2 Richardson–Lucy. Uses ImageJ-style argument syntax that works on Windows:
      -image file <path> -psf file <path>
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(java),
        "-jar",
        str(dl2_jar),
        "Run",
        "-image", "file", str(image_tif),
        "-psf", "file", str(psf_tif),
        "-algorithm", "RL", str(background), str(iterations),
        "-out", "mip", "MLI",
        "-path", str(out_dir),
    ]

    print("Running DL2:\n ", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n--- DL2 STDOUT ---\n", res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"DL2 failed.\n--- STDERR ---\n{res.stderr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--omezarr_dir",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\3d_data\20220218_dynamic\DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM\image.ome.zarr",
    )
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument(
        "--psf_tif",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\generated\psf_BW_TV1-T1-SR_Lambda405nm.tif",
    )
    ap.add_argument(
        "--out_dir",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\deconv",
    )
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

    # ensure DL2 jar exists 
    dl2_jar = ensure_deconvolutionlab2_exists(fiji_dir, auto_download=True, quiet=True)

    java = _find_java_exe(fiji_dir)

    omezarr_dir = Path(args.omezarr_dir)
    psf_path = Path(args.psf_tif)
    out_root = Path(args.out_dir)

    image_zyx = load_omezarr_channel_zyx(omezarr_dir, channel_index=args.channel)
    psf = tiff.imread(str(psf_path))
    basic_validity_checks(image_zyx, psf)

    print("Input summary (no changes applied):")
    print(f"  Image: shape={image_zyx.shape}, dtype={image_zyx.dtype}, min={image_zyx.min()}, max={image_zyx.max()}")
    print(f"  PSF  : shape={psf.shape}, dtype={psf.dtype}, min={psf.min()}, max={psf.max()}, sum={float(np.sum(psf))}")

    with tempfile.TemporaryDirectory(prefix="pft_dl2_") as td:
        td = Path(td)
        tmp_img = td / f"tmp_image_C{args.channel}.tif"
        tmp_psf = td / "tmp_psf.tif"

        # Use ImageJ-compatible stacks (still preserves uint16 / float32)
        write_imagej_tiff_stack(image_zyx, tmp_img, axes="ZYX")
        write_imagej_tiff_stack(psf, tmp_psf, axes="ZYX")

        out_dir = out_root / f"{omezarr_dir.parent.name}__C{args.channel}__iter{args.iters}"
        run_dl2_cli(
            java=java,
            dl2_jar=dl2_jar,
            image_tif=tmp_img,
            psf_tif=tmp_psf,
            out_dir=out_dir,
            iterations=args.iters,
            background=args.background,
        )

        print(f"\nDL2 finished. Outputs:\n  {out_dir}")

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