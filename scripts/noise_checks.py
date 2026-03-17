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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
try:
    import tifffile
except ImportError as e:
    raise ImportError("Please install tifffile: pip install tifffile") from e
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("Please install matplotlib: pip install matplotlib") from e
from PFT.core_prog_parts import visualization as viz
from PFT.core_prog_parts.common_paths import ensure_dir
try:
    import pandas as pd
except ImportError:
    pd = None


PATCH_ROOT = Path(
    r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\training_files\2d_wga_dapi\training_data\normvsnormalized_image"
)

CHANNEL_INDEX = 0
INCLUDE_ANY = ["NORMp1p99", "RAW"]
EXCLUDE_ANY = ["RAW_vs_NORMALIZED_RGB"]
SAVE_IMAGE_PNG = True
FFT_LOG_SCALE = True
EPS = 1e-12

def _as_float(img: np.ndarray) -> np.ndarray:
    """Convert to float32 without normalization."""
    return img.astype(np.float32, copy=False) if img.dtype != np.float32 else img


def pick_2d_view(arr: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """
    Convert array to a single 2D view for WHOLE-image FFT.

    Rules:
      - (Y,X) -> use as is
      - (C,Y,X) with small C -> take channel_index (CYX)
      - (N,Y,X) (stack)      -> take first slice (index 0)
      - (N,C,Y,X)            -> take [0, channel_index]
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        #(CYX)
        if arr.shape[0] <= 8:
            return arr[channel_index]
        return arr[0]

    if arr.ndim == 4:
        #(N,C,Y,X)
        if arr.shape[1] <= 8:
            return arr[0, channel_index]
        return arr[0, 0]

    raise ValueError(f"Unsupported array shape for FFT: ndim={arr.ndim}, shape={arr.shape}")


def neighbor_corr(img2d: np.ndarray) -> float:
    """Average Pearson correlation between pixel and its right/down neighbors."""
    img = _as_float(img2d)
    a = img[:, :-1].ravel()
    b = img[:, 1:].ravel()
    c = img[:-1, :].ravel()
    d = img[1:, :].ravel()

    def _corr(x, y) -> float:
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean())) + EPS
        return float((x * y).mean() / denom)

    return float(0.5 * (_corr(a, b) + _corr(c, d)))


def row_col_adj_corr(img2d: np.ndarray) -> Tuple[float, float]:
    """Correlation of adjacent row means / column means."""
    img = _as_float(img2d)
    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)

    def _adj_corr(v: np.ndarray) -> float:
        if v.size < 3:
            return 0.0
        x = v[:-1] - v[:-1].mean()
        y = v[1:] - v[1:].mean()
        denom = (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean())) + EPS
        return float((x * y).mean() / denom)

    return _adj_corr(row_means), _adj_corr(col_means)


def fft_peak_score(img2d: np.ndarray) -> float:
    img = _as_float(img2d)
    x = img - float(img.mean())
    F = np.fft.fftshift(np.fft.fft2(x))
    P = (np.abs(F) ** 2)

    h, w = P.shape
    cy, cx = h // 2, w // 2
    r = max(2, min(h, w) // 32)

    P_masked = P.copy()
    P_masked[cy - r: cy + r + 1, cx - r: cx + r + 1] = np.nan

    maxp = np.nanmax(P_masked)
    medp = np.nanmedian(P_masked) + EPS
    return float(maxp / medp)


def save_image_and_fft_png(
    img2d: np.ndarray,
    out_dir: Path,
    stem: str,
    log_scale: bool = True,
    save_image_png: bool = True,
) -> None:
    ensure_dir(out_dir)

    img = _as_float(img2d)
    x = img - float(img.mean())
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)
    disp = np.log1p(mag) if log_scale else mag

    viz.save_image_and_fft(img, disp, out_dir, stem, save_image_png=save_image_png)


@dataclass
class FileSummary:
    file: str
    shape_2d: str
    dtype: str
    mean: float
    var: float
    std: float
    fano: float
    neighbor_corr: float
    row_adj_corr: float
    col_adj_corr: float
    fft_peak_score: float


def list_tiff_files(root: Path) -> List[Path]:
    files = []
    files.extend(root.glob("*.tif"))
    files.extend(root.glob("*.tiff"))
    files.extend(root.glob("*.TIF"))
    files.extend(root.glob("*.TIFF"))
    return sorted(set(files))


def should_process(p: Path) -> bool:
    name = p.name
    if any(x in name for x in EXCLUDE_ANY):
        return False
    if not any(x in name for x in INCLUDE_ANY):
        return False
    return True


def write_summary_csv(path: Path, rows: List[FileSummary]) -> None:
    ensure_dir(path.parent)
    records = [asdict(r) for r in rows]
    if pd is not None:
        pd.DataFrame.from_records(records).to_csv(path, index=False)
    else:
        import csv
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)


def main() -> None:
    if not PATCH_ROOT.exists():
        raise FileNotFoundError(f"PATCH_ROOT does not exist: {PATCH_ROOT}")

    out_base = PATCH_ROOT / "_noise_diagnostics"
    out_fft_dir = out_base / "fft_single"
    out_csv = out_base / "fft_single_summary.csv"
    ensure_dir(out_base)

    files = [p for p in list_tiff_files(PATCH_ROOT) if should_process(p)]
    if not files:
        raise FileNotFoundError(
            f"No matching TIFFs found in {PATCH_ROOT}\n"
            f"Include any of: {INCLUDE_ANY}\n"
            f"Exclude any of: {EXCLUDE_ANY}"
        )

    summaries: List[FileSummary] = []
    for fp in files:
        arr = tifffile.imread(fp)
        img2d = pick_2d_view(arr, CHANNEL_INDEX)

        imgf = _as_float(img2d)
        m = float(imgf.mean())
        v = float(imgf.var())
        s = float(imgf.std())
        fano = float(v / (m + EPS))

        nc = neighbor_corr(imgf)
        rc, cc = row_col_adj_corr(imgf)
        fps = fft_peak_score(imgf)

        summaries.append(
            FileSummary(
                file=fp.name,
                shape_2d=str(tuple(img2d.shape)),
                dtype=str(img2d.dtype),
                mean=m,
                var=v,
                std=s,
                fano=fano,
                neighbor_corr=nc,
                row_adj_corr=rc,
                col_adj_corr=cc,
                fft_peak_score=fps,
            )
        )

        save_image_and_fft_png(
            img2d=img2d,
            out_dir=out_fft_dir,
            stem=fp.stem,
            log_scale=FFT_LOG_SCALE,
            save_image_png=SAVE_IMAGE_PNG,
        )

    write_summary_csv(out_csv, summaries)

    print(f"[DONE] Processed {len(files)} files")
    print(f"[DONE] FFT PNGs: {out_fft_dir}")
    print(f"[DONE] Summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
