r"""Provide command-line and programmatic utilities for Fourier-transform result three-dimensional data.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/FFT_3d.py --help

Representative execution:

    python scripts/denoising/FFT_3d.py \
        --root results/img/3d_data \
        --level 0 \
        --out results/example_output \
        --repo . \
        --time 0
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    The lookup is based on this script's physical location and therefore does
    not depend on the current working directory. An explicit error is raised
    when the expected repository layout cannot be found.

    Args:
        start (_PFTPath | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        _PFTPath: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _pft_project_root()
    """
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tifffile as tiff


"Explores frequency patterns in 3D image data."

def _import_pft_decoder(repo_root: Path):
    """Import the canonical decoder that returns data in ``(C, Z, Y, X)`` order.

    The ``repo_root`` argument is retained for compatibility with existing
    callers. Repository imports are configured once at module startup.

    Args:
        repo_root (Path): Directory used for repo.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = _import_pft_decoder(repo_root=Path("path/to/resource"))
    """
    del repo_root
    from PFT.core_prog_parts.decoder_omezar import decode_omezarr_volume

    return decode_omezarr_volume


def find_omezarr_store(sample_dir: Path) -> Optional[Path]:
    """Find OME-Zarr store in the available data or project structure.

    Args:
        sample_dir (Path): Directory used for sample.

    Returns:
        Optional[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = find_omezarr_store(sample_dir=Path("path/to/resource"))
    """
    preferred = sample_dir / "image.ome.zarr"
    if preferred.is_dir():
        return preferred
    for p in sample_dir.iterdir():
        if p.is_dir() and p.name.lower().endswith(".ome.zarr"):
            return p
    for p in sample_dir.iterdir():
        if p.is_dir() and p.name.lower().endswith(".zarr"):
            return p
    return None


def pick_evenly_spaced_slices(s0: int, s1: int, n: int) -> List[int]:
    """Return pick evenly spaced slices for the supplied inputs.

    Args:
        s0 (int): Numerical value controlling s0.
        s1 (int): Numerical value controlling s1.
        n (int): Numerical value controlling n.

    Returns:
        List[int]: Collection containing the generated or selected values.

    Example:
        >>> result = pick_evenly_spaced_slices(
        ...     s0=1,
        ...     s1=1,
        ...     n=1,
        ... )
    """
    if s1 < s0:
        s0, s1 = s1, s0
    if s0 == s1:
        return [s0]
    length = s1 - s0 + 1
    if length <= n:
        return list(range(s0, s1 + 1))

    vals = np.linspace(s0, s1, n)
    idx = np.round(vals).astype(int)
    idx = np.clip(idx, s0, s1)

    out: List[int] = []
    seen = set()
    for i in idx.tolist():
        if i not in seen:
            out.append(i)
            seen.add(i)

    if len(out) < n:
        for k in range(s0, s1 + 1):
            if k not in seen:
                out.append(k)
                seen.add(k)
            if len(out) == n:
                break

    return out[:n]


def percentile_norm_uint16(img: np.ndarray, p_low=1.0, p_high=99.8) -> np.ndarray:
    """Return percentile norm uint16 for the supplied inputs.

    Args:
        img (np.ndarray): Array containing img.
        p_low (Any): Value specifying p low for the operation. Defaults to ``1.0``.
        p_high (Any): Value specifying p high for the operation. Defaults to ``99.8``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = percentile_norm_uint16(img=image_array)
    """
    x = img.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(img, dtype=np.uint16)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 65535.0 + 0.5).astype(np.uint16)


def rgb_composite_uint16(c0, c1, c2) -> np.ndarray:
    # R=c2, G=c1, B=c0
    """Return RGB representation composite uint16 for the supplied inputs.

    Args:
        c0 (Any): Value specifying c0 for the operation.
        c1 (Any): Value specifying c1 for the operation.
        c2 (Any): Value specifying c2 for the operation.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = rgb_composite_uint16(
        ...     c0=...,
        ...     c1=...,
        ...     c2=...,
        ... )
    """
    return np.stack([c2, c1, c0], axis=-1).astype(np.uint16, copy=False)


def fft_image_uint16(img: np.ndarray) -> np.ndarray:
    """Return Fourier-transform result image uint16 for the supplied inputs.

    Args:
        img (np.ndarray): Array containing img.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = fft_image_uint16(img=image_array)
    """
    x = img.astype(np.float32, copy=False)
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.log1p(np.abs(F))
    mn, mx = float(mag.min()), float(mag.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(img, dtype=np.uint16)
    mag = (mag - mn) / (mx - mn)
    return (mag * 65535.0 + 0.5).astype(np.uint16)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return safe corr for the supplied inputs.

    Args:
        a (np.ndarray): Array containing a.
        b (np.ndarray): Array containing b.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = _safe_corr(a=image_array, b=image_array)
    """
    a = a.ravel().astype(np.float32, copy=False)
    b = b.ravel().astype(np.float32, copy=False)
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0


def neighbor_corr(img: np.ndarray) -> float:
    """Return neighbor corr for the supplied inputs.

    Args:
        img (np.ndarray): Array containing img.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = neighbor_corr(img=image_array)
    """
    if img.shape[0] < 2 or img.shape[1] < 2:
        return 0.0
    return 0.5 * (
        _safe_corr(img[:-1, :-1], img[:-1, 1:])
        + _safe_corr(img[:-1, :-1], img[1:, :-1])
    )


def fft_peak_score(img: np.ndarray) -> float:
    """Return Fourier-transform result peak score for the supplied inputs.

    Args:
        img (np.ndarray): Array containing img.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = fft_peak_score(img=image_array)
    """
    x = img.astype(np.float32, copy=False)
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = max(3, min(h, w) // 128)
    mag[cy - r : cy + r + 1, cx - r : cx + r + 1] = 0.0
    med = float(np.median(mag))
    mx = float(np.max(mag))
    if not np.isfinite(mx) or not np.isfinite(med) or med <= 1e-12:
        return 0.0
    return mx / med


def fft_directionality(img: np.ndarray) -> Tuple[float, float]:
    """Returns: score in [0..1] (higher => more directional/anisotropic in frequency domain), angle_deg (dominant direction, 180°-periodic).

    Args:
        img (np.ndarray): Array containing img.

    Returns:
        Tuple[float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = fft_directionality(img=image_array)
    """
    x = img.astype(np.float32, copy=False)
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.indices((h, w))
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dx * dx + dy * dy)

    r0 = max(5, min(h, w) // 128)
    mask = rr >= r0
    if not np.any(mask):
        return 0.0, 0.0

    theta = np.arctan2(dy[mask], dx[mask])
    wgt = mag[mask].astype(np.float64, copy=False)
    s = float(np.sum(wgt))
    if not np.isfinite(s) or s <= 1e-12:
        return 0.0, 0.0

    vec = np.sum(wgt * np.exp(2j * theta))  # 180° symmetry
    score = float(np.abs(vec) / s)
    ang = float(0.5 * np.angle(vec) * 180.0 / math.pi)

    if not np.isfinite(score):
        score = 0.0
    if not np.isfinite(ang):
        ang = 0.0
    return score, ang


def _get_channel_safe(vol_czyx: np.ndarray, c: int) -> np.ndarray:
    """Return vol[c] if exists else zeros with shape (Z,Y,X).

    Args:
        vol_czyx (np.ndarray): Array containing vol czyx.
        c (int): Numerical value controlling c.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _get_channel_safe(vol_czyx=image_array, c=1)
    """
    C, Z, H, W = vol_czyx.shape
    if 0 <= c < C:
        return vol_czyx[c]
    return np.zeros((Z, H, W), dtype=vol_czyx.dtype)


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--root",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\3d_data",
        help="Root folder containing 3d_data datasets.",
    )
    parser.add_argument(
        "--out",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\fft\3d_data",
        help="Output folder for FFT results + CSV.",
    )
    parser.add_argument(
        "--repo",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT",
        help="Repo root (used to import PFT decoder).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="OME-Zarr pyramid level (0=full res).",
    )

    
    parser.add_argument(
        "--time",
        type=int,
        default=0,
        help="Time index to select if 't' axis exists.",
    )

    parser.add_argument(
        "--z_start",
        type=int,
        default=20,
        help="First Z slice index for preview export.",
    )
    parser.add_argument(
        "--z_end",
        type=int,
        default=30,
        help="Last Z slice index for preview export.",
    )
    parser.add_argument(
        "--n_vis",
        type=int,
        default=5,
        help="How many preview Z slices to export between z_start and z_end.",
    )

    
    parser.add_argument("--p_low", type=float, default=1.0)
    parser.add_argument("--p_high", type=float, default=99.8)

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    decode_omezarr_volume = _import_pft_decoder(Path(args.repo))

    rows: List[Dict[str, object]] = []

    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for sample_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            zarr_dir = find_omezarr_store(sample_dir)
            if zarr_dir is None:
                continue

            print(f"\nProcessing: {dataset_dir.name} / {sample_dir.name}")
            print(f"  Zarr: {zarr_dir}")

            # (C,Z,Y,X)
            vol, meta = decode_omezarr_volume(
                zarr_dir,
                level=int(args.level),
                time=int(args.time),
                channels=None,          # load all channels; we export B,G,R safely below
                as_float32=False,
                print_info=True,
            )

            C, Z, H, W = vol.shape
            print(f"  Loaded (C,Z,Y,X)={vol.shape}, dtype={vol.dtype}")

            ch_labels = ["B", "G", "R"]
            for z in range(Z):
                for ci, ch_name in enumerate(ch_labels):
                    img = _get_channel_safe(vol, ci)[z].astype(np.float32, copy=False)  # (Y,X)

                    mean = float(np.mean(img))
                    var = float(np.var(img))
                    std = float(np.std(img))
                    fano = float(var / mean) if abs(mean) > 1e-12 else 0.0
                    dscore, dang = fft_directionality(img)

                    rows.append(
                        {
                            "dataset": dataset_dir.name,
                            "sample": sample_dir.name,
                            "file": str(zarr_dir),
                            "slice_axis": "z",
                            "slice_index": z,
                            "channel": ch_name,
                            "height": H,
                            "width": W,
                            "mean": mean,
                            "var": var,
                            "std": std,
                            "fano": fano,
                            "neighbor_corr": neighbor_corr(img),
                            "fft_peak_score": fft_peak_score(img),
                            "fft_dir_score": dscore,
                            "fft_dir_angle_deg": dang,
                        }
                    )

            if Z <= 1:
                z_list = [0]
            else:
                z0 = max(0, min(int(args.z_start), Z - 1))
                z1 = max(0, min(int(args.z_end), Z - 1))
                z_list = pick_evenly_spaced_slices(z0, z1, int(args.n_vis))

            print(f"  Visualize z-slices: {z_list}")

            sample_out = out_root / dataset_dir.name / sample_dir.name
            sample_out.mkdir(parents=True, exist_ok=True)

            c0_vol = _get_channel_safe(vol, 0)
            c1_vol = _get_channel_safe(vol, 1)
            c2_vol = _get_channel_safe(vol, 2)

            for z in z_list:
                c0 = c0_vol[z]
                c1 = c1_vol[z]
                c2 = c2_vol[z]

                # RAW -> uint16
                c0_raw = c0.astype(np.uint16, copy=False) if c0.dtype != np.uint16 else c0
                c1_raw = c1.astype(np.uint16, copy=False) if c1.dtype != np.uint16 else c1
                c2_raw = c2.astype(np.uint16, copy=False) if c2.dtype != np.uint16 else c2

                rgb_raw = rgb_composite_uint16(c0_raw, c1_raw, c2_raw)
                tiff.imwrite(sample_out / f"z{z:03d}_RGB_RAW_uint16.tif", rgb_raw, photometric="rgb")

                # NORM (per-channel)
                c0n = percentile_norm_uint16(c0_raw, args.p_low, args.p_high)
                c1n = percentile_norm_uint16(c1_raw, args.p_low, args.p_high)
                c2n = percentile_norm_uint16(c2_raw, args.p_low, args.p_high)

                rgb_norm = rgb_composite_uint16(c0n, c1n, c2n)
                tiff.imwrite(
                    sample_out / f"z{z:03d}_RGB_NORMp{args.p_low:g}p{args.p_high:g}_uint16.tif",
                    rgb_norm,
                    photometric="rgb",
                )

                # FFTs (RAW + NORM) per channel
                tiff.imwrite(sample_out / f"z{z:03d}_FFT_RAW_B.tif", fft_image_uint16(c0_raw))
                tiff.imwrite(sample_out / f"z{z:03d}_FFT_RAW_G.tif", fft_image_uint16(c1_raw))
                tiff.imwrite(sample_out / f"z{z:03d}_FFT_RAW_R.tif", fft_image_uint16(c2_raw))

                tiff.imwrite(sample_out / f"z{z:03d}_FFT_NORM_B.tif", fft_image_uint16(c0n))
                tiff.imwrite(sample_out / f"z{z:03d}_FFT_NORM_G.tif", fft_image_uint16(c1n))
                tiff.imwrite(sample_out / f"z{z:03d}_FFT_NORM_R.tif", fft_image_uint16(c2n))

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = out_root / "fft_metrics_3d_slices.csv"
    df.to_csv(csv_path, index=False)

    print("\nDONE")
    print(f"Saved CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())