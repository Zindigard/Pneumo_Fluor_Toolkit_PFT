from __future__ import annotations

from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
for _p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (_p / "src" / "PFT").exists():
        _SRC_DIR = _p / "src"
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        break

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root
import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import tifffile as tiff

"Prepares and normalizes PSF data"


@dataclass
class PSFNormResult:
    file: str
    shape: str
    dtype: str
    min: float
    max: float
    sum: float
    sum_ok: bool
    max_ok: bool
    has_negative: bool
    com_z: float
    com_y: float
    com_x: float
    notes: str


def _load_psf(path: Path) -> np.ndarray:
    with tiff.TiffFile(str(path)) as tf:
        arr = tf.asarray()
    return arr


def _center_of_mass(a: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute intensity-weighted center of mass for 3D arrays.
    Assumes array order is (z,y,x) or (y,x) -> will be handled.
    """
    af = a.astype(np.float64, copy=False)
    total = float(af.sum())
    if total <= 0:
        return (float("nan"), float("nan"), float("nan"))

    if af.ndim == 2:
        y, x = np.indices(af.shape)
        com_y = float((af * y).sum() / total)
        com_x = float((af * x).sum() / total)
        return (float("nan"), com_y, com_x)

    if af.ndim != 3:
        return (float("nan"), float("nan"), float("nan"))

    z, y, x = np.indices(af.shape)
    com_z = float((af * z).sum() / total)
    com_y = float((af * y).sum() / total)
    com_x = float((af * x).sum() / total)
    return (com_z, com_y, com_x)


def analyze_psf(
    path: Path,
    sum_target: float = 1.0,
    sum_tol: float = 1e-3,
    max_target: float = 1.0,
    max_tol: float = 1e-3,
) -> PSFNormResult:
    a = _load_psf(path)
    af = a.astype(np.float64, copy=False)

    mn = float(np.min(af))
    mx = float(np.max(af))
    sm = float(np.sum(af))

    has_negative = mn < 0

    sum_ok = (abs(sm - sum_target) <= sum_tol) if np.isfinite(sm) else False
    max_ok = (abs(mx - max_target) <= max_tol) if np.isfinite(mx) else False

    com_z, com_y, com_x = _center_of_mass(af)

    notes = []
    if has_negative:
        notes.append("NEGATIVE_VALUES")
    if sm == 0:
        notes.append("SUM_ZERO")
    if not sum_ok:
        notes.append("SUM_NOT_1")
    if not max_ok:
        notes.append("MAX_NOT_1")

    return PSFNormResult(
        file=path.name,
        shape=str(tuple(a.shape)),
        dtype=str(a.dtype),
        min=mn,
        max=mx,
        sum=sm,
        sum_ok=sum_ok,
        max_ok=max_ok,
        has_negative=has_negative,
        com_z=com_z,
        com_y=com_y,
        com_x=com_x,
        notes=";".join(notes) if notes else "OK",
    )


def normalize_psf(a: np.ndarray, mode: str) -> np.ndarray:
    af = a.astype(np.float64, copy=False)
    if mode == "sum1":
        s = float(af.sum())
        if s == 0:
            raise ValueError("Cannot normalize by sum: sum is zero.")
        out = af / s
        return out
    if mode == "max1":
        m = float(af.max())
        if m == 0:
            raise ValueError("Cannot normalize by max: max is zero.")
        out = af / m
        return out
    raise ValueError(f"Unknown mode: {mode}")


def save_like_imagej(src_path: Path, out_path: Path, arr_float: np.ndarray) -> None:
    """
    Preserve ImageJ metadata style by copying ImageDescription if present.
    We write float32 output to keep normalized values.
    """
    with tiff.TiffFile(str(src_path)) as tf:
        tags = tf.pages[0].tags
        desc = tags.get("ImageDescription").value if "ImageDescription" in tags else None
        xres = tags.get("XResolution").value if "XResolution" in tags else None
        yres = tags.get("YResolution").value if "YResolution" in tags else None
        resunit = tags.get("ResolutionUnit").value if "ResolutionUnit" in tags else None

    kwargs = {}
    if isinstance(desc, (str, bytes)):
        kwargs["description"] = desc
    if xres is not None and yres is not None:
        kwargs["resolution"] = (xres, yres)
    if resunit is not None:
        kwargs["resolutionunit"] = resunit

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), arr_float.astype(np.float32), imagej=True, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check (and optionally normalize) generated PSF TIFFs.")
    ap.add_argument(
        "--generated_dir",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\generated",
        help="Folder containing psf_*.tif files.",
    )
    ap.add_argument(
        "--pattern",
        default="psf_*.tif",
        help="Glob pattern for PSF files (default: psf_*.tif).",
    )
    ap.add_argument(
        "--out_csv",
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\psf_norm_report.csv",
        help="CSV report output.",
    )
    ap.add_argument(
        "--sum_tol",
        type=float,
        default=1e-3,
        help="Tolerance for sum≈1 check (default 1e-3).",
    )
    ap.add_argument(
        "--max_tol",
        type=float,
        default=1e-3,
        help="Tolerance for max≈1 check (default 1e-3).",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="If set, write normalized copies for PSFs that are not normalized.",
    )
    ap.add_argument(
        "--mode",
        choices=["sum1", "max1"],
        default="sum1",
        help="Normalization mode if --normalize is set: sum1 (energy) or max1 (peak).",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to write normalized PSFs (default: <generated_dir>/normalized_<mode>).",
    )
    args = ap.parse_args()

    generated_dir = Path(args.generated_dir)
    if not generated_dir.exists():
        raise FileNotFoundError(generated_dir)

    files = sorted(generated_dir.glob(args.pattern))
    if not files:
        print(f"No files found in {generated_dir} with pattern {args.pattern}")
        return 0

    results: List[PSFNormResult] = []

    out_dir = Path(args.out_dir) if args.out_dir else (generated_dir / f"normalized_{args.mode}")

    for p in files:
        r = analyze_psf(p, sum_tol=args.sum_tol, max_tol=args.max_tol)
        results.append(r)

        if args.normalize:
            need = (not r.sum_ok) if args.mode == "sum1" else (not r.max_ok)
            if need and np.isfinite(r.sum) and r.sum != 0 and np.isfinite(r.max) and r.max != 0:
                a = _load_psf(p)
                arrn = normalize_psf(a, args.mode)
                out_path = out_dir / p.name.replace(".tif", f"__{args.mode}.tif")
                save_like_imagej(p, out_path, arrn)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    n = len(results)
    sum_ok = sum(1 for r in results if r.sum_ok)
    max_ok = sum(1 for r in results if r.max_ok)
    neg = sum(1 for r in results if r.has_negative)

    print(f"Checked {n} PSFs in: {generated_dir}")
    print(f"Sum≈1 OK : {sum_ok}/{n}  (tol={args.sum_tol})")
    print(f"Max≈1 OK : {max_ok}/{n}  (tol={args.max_tol})")
    print(f"Negative : {neg}/{n}")
    print(f"CSV report: {out_csv}")
    if args.normalize:
        print(f"Normalized outputs (mode={args.mode}) in: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())