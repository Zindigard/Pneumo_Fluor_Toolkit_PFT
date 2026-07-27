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
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
try:
    import tifffile as tiff
except ImportError as e:
    raise SystemExit("Missing dependency: tifffile. Install with: pip install tifffile") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Missing dependency: pillow. Install with: pip install pillow") from e

"Explores frequency patterns in 2D images"
EPS = 1e-12


def _save_png(path: Path, img_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img_u8.ndim == 2:
        Image.fromarray(img_u8, mode="L").save(path)
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        Image.fromarray(img_u8, mode="RGB").save(path)
    else:
        Image.fromarray(img_u8[..., 0], mode="L").save(path)


def _fft_vis_2d(x2d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x2d.astype(np.float32, copy=False)
    x = x - float(np.mean(x))

    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    mag = np.abs(F)

    vis = np.log1p(mag + eps)
    lo = np.percentile(vis, 1.0)
    hi = np.percentile(vis, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    vis = (vis - lo) / (hi - lo)
    vis = np.clip(vis, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS
    return float(np.sum(a * b) / denom)


def _mean_adjacent_corr_rows(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    corrs = [_pearson_corr(x[i, :], x[i + 1, :]) for i in range(x.shape[0] - 1)]
    return float(np.mean(corrs)) if corrs else 0.0


def _mean_adjacent_corr_cols(x: np.ndarray) -> float:
    if x.shape[1] < 2:
        return 0.0
    corrs = [_pearson_corr(x[:, j], x[:, j + 1]) for j in range(x.shape[1] - 1)]
    return float(np.mean(corrs)) if corrs else 0.0


def _neighbor_corr(x: np.ndarray) -> float:
    vals = []
    if x.shape[1] >= 2:
        vals.append(_pearson_corr(x[:, :-1], x[:, 1:]))
    if x.shape[0] >= 2:
        vals.append(_pearson_corr(x[:-1, :], x[1:, :]))
    return float(np.mean(vals)) if vals else 0.0


def _fft_peak_score(x: np.ndarray, dc_halfwidth: int = 8) -> float:
    x = x.astype(np.float32, copy=False)
    x = x - float(np.mean(x))

    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    mag = np.abs(F).astype(np.float64)

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    mask = np.ones_like(mag, dtype=bool)
    y0 = max(0, cy - dc_halfwidth)
    y1 = min(h, cy + dc_halfwidth + 1)
    x0 = max(0, cx - dc_halfwidth)
    x1 = min(w, cx + dc_halfwidth + 1)
    mask[y0:y1, x0:x1] = False

    mag2 = mag[mask]
    if mag2.size == 0:
        return 0.0

    peak = float(np.max(mag2))
    baseline = float(np.median(mag2)) + EPS
    return peak / baseline


def _fft_directionality(x: np.ndarray, dc_halfwidth: int = 8) -> tuple[float, float]:
    """
    Returns:
      (dir_score, dir_angle_deg)

    dir_score: anisotropy of FFT power second moment in [0,1]
    dir_angle_deg: dominant frequency direction angle in degrees, in [-90, 90)
                   (angle in frequency plane; stripes in image are perpendicular)
    """
    x = x.astype(np.float32, copy=False)
    x = x - float(np.mean(x))

    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    P = (np.abs(F) ** 2).astype(np.float64)

    h, w = P.shape
    cy, cx = h // 2, w // 2

    mask = np.ones_like(P, dtype=bool)
    y0 = max(0, cy - dc_halfwidth)
    y1 = min(h, cy + dc_halfwidth + 1)
    x0 = max(0, cx - dc_halfwidth)
    x1 = min(w, cx + dc_halfwidth + 1)
    mask[y0:y1, x0:x1] = False

    Pm = P * mask
    s = float(Pm.sum())
    if s <= 0:
        return 0.0, 0.0

    yy, xx = np.indices((h, w))
    u = (xx - cx).astype(np.float64)
    v = (yy - cy).astype(np.float64)

    Mxx = float((Pm * (u * u)).sum() / s)
    Myy = float((Pm * (v * v)).sum() / s)
    Mxy = float((Pm * (u * v)).sum() / s)

    tr = Mxx + Myy
    det = Mxx * Myy - Mxy * Mxy
    disc = max(tr * tr - 4.0 * det, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    lam1 = 0.5 * (tr + sqrt_disc)
    lam2 = 0.5 * (tr - sqrt_disc)

    denom = lam1 + lam2 + EPS
    dir_score = float((lam1 - lam2) / denom)
    dir_score = float(np.clip(dir_score, 0.0, 1.0))

    theta = 0.5 * np.arctan2(2.0 * Mxy, (Mxx - Myy))
    dir_angle_deg = float(np.degrees(theta))

    if dir_angle_deg >= 90.0:
        dir_angle_deg -= 180.0
    if dir_angle_deg < -90.0:
        dir_angle_deg += 180.0

    return dir_score, dir_angle_deg


def _to_2d_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32, copy=False)
        return (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).astype(np.float32)

    x = arr.astype(np.float32, copy=False)
    while x.ndim > 2:
        x = x.mean(axis=0)
    return x.astype(np.float32, copy=False)


@dataclass
class MetricsRow:
    dataset: str
    file: str
    channel: str
    height: int
    width: int
    mean: float
    var: float
    std: float
    fano: float
    neighbor_corr: float
    row_adj_corr: float
    col_adj_corr: float
    fft_peak_score: float
    fft_dir_score: float
    fft_dir_angle_deg: float


def compute_metrics(x2d: np.ndarray) -> dict[str, float]:
    x = x2d.astype(np.float64, copy=False)

    mean = float(np.mean(x))
    var = float(np.var(x))
    std = float(np.sqrt(var))
    fano = float(var / (mean + EPS)) if abs(mean) > 1e-12 else 0.0

    dir_score, dir_angle = _fft_directionality(x)

    return {
        "mean": mean,
        "var": var,
        "std": std,
        "fano": fano,
        "neighbor_corr": _neighbor_corr(x),
        "row_adj_corr": _mean_adjacent_corr_rows(x),
        "col_adj_corr": _mean_adjacent_corr_cols(x),
        "fft_peak_score": _fft_peak_score(x),
        "fft_dir_score": dir_score,
        "fft_dir_angle_deg": dir_angle,
    }


def iter_tiffs(root: Path, name_patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pat in name_patterns:
        files.extend(root.rglob(pat))
    return sorted(set(files))


def process_file(
    dataset_name: str,
    tif_path: Path,
    out_root_for_dataset: Path,
    base_root: Path,
    rows: list[MetricsRow],
) -> bool:
    try:
        img = tiff.imread(str(tif_path))
    except Exception as e:
        print(f"[SKIP] Could not read: {tif_path} ({e})")
        return False

    rel = tif_path.relative_to(base_root)
    out_path = (out_root_for_dataset / rel).with_suffix("")
    out_png_base = out_path.parent / out_path.name

    arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32, copy=False)

        fft_rgb = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            fft_rgb[..., c] = _fft_vis_2d(rgb[..., c])
        _save_png(out_png_base.parent / f"{out_png_base.name}_fft.png", fft_rgb)

        ch_names = ["R", "G", "B"]
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        for c, ch in enumerate(ch_names):
            m = compute_metrics(rgb[..., c])
            rows.append(
                MetricsRow(
                    dataset=dataset_name,
                    file=str(tif_path),
                    channel=ch,
                    height=h,
                    width=w,
                    **m,
                )
            )

        print(f"[OK] [{dataset_name}] {tif_path} -> FFT PNG + metrics (R/G/B)")
        return True

    x2d = _to_2d_float(arr)
    fft_u8 = _fft_vis_2d(x2d)
    _save_png(out_png_base.parent / f"{out_png_base.name}_fft.png", fft_u8)

    m = compute_metrics(x2d)
    h, w = int(x2d.shape[0]), int(x2d.shape[1])
    rows.append(
        MetricsRow(
            dataset=dataset_name,
            file=str(tif_path),
            channel="gray",
            height=h,
            width=w,
            **m,
        )
    )

    print(f"[OK] [{dataset_name}] {tif_path} -> FFT PNG + metrics")
    return True


def write_csv(csv_path: Path, rows: Iterable[MetricsRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "file",
        "channel",
        "height",
        "width",
        "mean",
        "var",
        "std",
        "fano",
        "neighbor_corr",
        "row_adj_corr",
        "col_adj_corr",
        "fft_peak_score",
        "fft_dir_score",
        "fft_dir_angle_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch 2D FFT diagnostic + metrics CSV (incl. directionality) for TIFF images across multiple datasets."
    )
    ap.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        default=[
            Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\2d_time"),
            Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\2d_wga_dapi"),
        ],
        help="One or more dataset roots to scan.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\fft"),
        help="Where FFT PNGs and CSV will be written (default: results/fft).",
    )
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["image_norm16_rgb*.tif", "image_norm16_rgb*.tiff", "image_raw_rgb*.tif", "image_raw_rgb*.tiff"],
        help="Filename patterns to include.",
    )
    ap.add_argument(
        "--csv-name",
        type=str,
        default="fft_metrics.csv",
        help="CSV filename (written under output-root).",
    )
    args = ap.parse_args()

    out_root: Path = args.output_root
    patterns = tuple(args.patterns)
    csv_path = out_root / args.csv_name

    rows: list[MetricsRow] = []
    total_found = 0
    total_ok = 0

    for in_root in args.input_roots:
        if not in_root.exists():
            print(f"[WARN] Input root does not exist, skipping: {in_root}")
            continue

        dataset_name = in_root.name  
        out_root_for_dataset = out_root / dataset_name

        tiffs = iter_tiffs(in_root, patterns)
        total_found += len(tiffs)

        print(f"\n=== DATASET: {dataset_name} ===")
        print(f"Input root:  {in_root}")
        print(f"Output root: {out_root_for_dataset}")
        print(f"Found {len(tiffs)} TIFF(s) matching patterns: {patterns}")

        ok = 0
        for p in tiffs:
            ok += int(
                process_file(
                    dataset_name=dataset_name,
                    tif_path=p,
                    out_root_for_dataset=out_root_for_dataset,
                    base_root=in_root,
                    rows=rows,
                )
            )
        total_ok += ok

    write_csv(csv_path, rows)
    print("\n=== SUMMARY ===")
    print(f"Total TIFFs found: {total_found}")
    print(f"Total processed : {total_ok}")
    print(f"CSV written     : {csv_path}")
    print(f"Rows in CSV     : {len(rows)} (includes per-channel rows for RGB images)")


if __name__ == "__main__":
    main()