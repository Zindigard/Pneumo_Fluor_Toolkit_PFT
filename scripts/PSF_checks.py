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
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import tifffile as tiff

"Checks PSF files to make sure they are valid"

def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tiff_info(path: Path) -> dict:
    info: dict = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "md5": md5sum(path),
    }

    with tiff.TiffFile(str(path)) as tf:
        info["pages"] = len(tf.pages)
        info["series"] = len(tf.series)

        arr = tf.asarray()
        info["shape"] = tuple(arr.shape)
        info["dtype"] = str(arr.dtype)

        a = arr.astype(np.float64, copy=False)
        info["min"] = float(np.min(a))
        info["max"] = float(np.max(a))
        info["mean"] = float(np.mean(a))
        info["std"] = float(np.std(a))

        tags = tf.pages[0].tags
        info["XResolution"] = tags.get("XResolution").value if "XResolution" in tags else None
        info["YResolution"] = tags.get("YResolution").value if "YResolution" in tags else None
        info["ResolutionUnit"] = tags.get("ResolutionUnit").value if "ResolutionUnit" in tags else None

        desc = tags.get("ImageDescription").value if "ImageDescription" in tags else None
        if isinstance(desc, bytes):
            try:
                desc = desc.decode("utf-8", errors="replace")
            except Exception:
                desc = str(desc)
        info["ImageDescription_snip"] = (desc[:800] + "…") if isinstance(desc, str) and len(desc) > 800 else desc

        spacing = None
        if isinstance(desc, str):
            m = re.search(r"spacing=([0-9]*\.?[0-9]+)", desc)
            if m:
                try:
                    spacing = float(m.group(1))
                except Exception:
                    spacing = None
        info["ImageJ_spacing"] = spacing

    return info


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    out: dict = {
        "same_shape": a.shape == b.shape,
        "same_dtype": str(a.dtype) == str(b.dtype),
    }
    if a.shape != b.shape:
        return out

    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    diff = af - bf

    out["max_abs_diff"] = float(np.max(np.abs(diff)))
    out["MAE"] = float(np.mean(np.abs(diff)))
    out["MSE"] = float(np.mean(diff ** 2))

    a_std = float(np.std(af))
    b_std = float(np.std(bf))
    if a_std > 0 and b_std > 0:
        out["pearson_corr"] = float(np.corrcoef(af.ravel(), bf.ravel())[0, 1])
    else:
        out["pearson_corr"] = None

    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        out["n_pixels_different_exact"] = int(np.count_nonzero(a != b))
    else:
        out["n_pixels_different_exact"] = None

    return out


@dataclass
class Pair:
    model: str
    color: str               # Blue/Green/Red
    wavelength_nm: int
    imagej_path: Path
    ours_path: Path


def discover_pairs(generated_dir: Path) -> List[Pair]:
    """
    Matches:
      ImageJ:  "PSF <MODEL> <Color>.tif"   e.g. "PSF BW Blue.tif"
      Ours:    "psf_<MODEL>_*_Lambda<NNN>nm.tif" e.g. "psf_BW_TV1-T1-SR_Lambda405nm.tif"

    Color ↔ wavelength mapping (adjust if your red isn't 561):
      Blue  -> 405
      Green -> 488
      Red   -> 561
    """
    color_to_lambda = {"Blue": 405, "Green": 488, "Red": 561}

    imagej_re = re.compile(r"^PSF\s+(BW|GL|RW)\s+(Blue|Green|Red)\.tif$", re.IGNORECASE)
    imagej: Dict[Tuple[str, str], Path] = {}
    for p in generated_dir.glob("*.tif"):
        m = imagej_re.match(p.name)
        if m:
            model = m.group(1).upper()
            color = m.group(2).capitalize()
            imagej[(model, color)] = p

    ours_re = re.compile(r"^psf_(BW|GL|RW)_.+_Lambda(\d+)nm\.tif$", re.IGNORECASE)
    ours: Dict[Tuple[str, int], Path] = {}
    for p in generated_dir.glob("psf_*.tif"):
        m = ours_re.match(p.name)
        if m:
            model = m.group(1).upper()
            lam = int(m.group(2))
            ours[(model, lam)] = p

    pairs: List[Pair] = []
    for (model, color), imgj_path in sorted(imagej.items()):
        lam = color_to_lambda[color]
        ours_path = ours.get((model, lam))
        if ours_path is None:
            continue
        pairs.append(Pair(model=model, color=color, wavelength_nm=lam,
                          imagej_path=imgj_path, ours_path=ours_path))

    return pairs



def write_reports(pairs: List[Pair], out_txt: Path, out_csv: Path) -> None:
    lines: List[str] = []
    rows: List[dict] = []

    lines.append("PSF BATCH COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append(f"Pairs found: {len(pairs)}")
    lines.append("")

    for i, pair in enumerate(pairs, 1):
        lines.append("-" * 60)
        lines.append(f"{i}) MODEL={pair.model} | COLOR={pair.color} | Lambda={pair.wavelength_nm} nm")
        lines.append(f"   ImageJ: {pair.imagej_path.name}")
        lines.append(f"   Ours  : {pair.ours_path.name}")
        lines.append("")

        info_a = read_tiff_info(pair.imagej_path)
        info_b = read_tiff_info(pair.ours_path)

        with tiff.TiffFile(str(pair.imagej_path)) as tf:
            arr_a = tf.asarray()
        with tiff.TiffFile(str(pair.ours_path)) as tf:
            arr_b = tf.asarray()

        comp = compare_arrays(arr_a, arr_b)

        lines.append("A (ImageJ) TIFF INFO")
        for k in ["path", "size_bytes", "md5", "pages", "series", "shape", "dtype", "min", "max", "mean", "std",
                  "XResolution", "YResolution", "ResolutionUnit", "ImageJ_spacing"]:
            lines.append(f"  {k}: {info_a.get(k)}")
        lines.append("  ImageDescription_snip:")
        lines.append(f"    {info_a.get('ImageDescription_snip')}")
        lines.append("")

        lines.append("B (Ours) TIFF INFO")
        for k in ["path", "size_bytes", "md5", "pages", "series", "shape", "dtype", "min", "max", "mean", "std",
                  "XResolution", "YResolution", "ResolutionUnit", "ImageJ_spacing"]:
            lines.append(f"  {k}: {info_b.get(k)}")
        lines.append("  ImageDescription_snip:")
        lines.append(f"    {info_b.get('ImageDescription_snip')}")
        lines.append("")

        lines.append("COMPARISON")
        for k, v in comp.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

        row = {
            "model": pair.model,
            "color": pair.color,
            "lambda_nm": pair.wavelength_nm,
            "imagej_file": pair.imagej_path.name,
            "ours_file": pair.ours_path.name,
            "same_shape": comp.get("same_shape"),
            "same_dtype": comp.get("same_dtype"),
            "MAE": comp.get("MAE"),
            "MSE": comp.get("MSE"),
            "max_abs_diff": comp.get("max_abs_diff"),
            "pearson_corr": comp.get("pearson_corr"),
            "n_pixels_different_exact": comp.get("n_pixels_different_exact"),
            # handy calibration checks
            "imagej_spacing": info_a.get("ImageJ_spacing"),
            "ours_spacing": info_b.get("ImageJ_spacing"),
            "imagej_shape": info_a.get("shape"),
            "ours_shape": info_b.get("shape"),
            "imagej_dtype": info_a.get("dtype"),
            "ours_dtype": info_b.get("dtype"),
        }
        rows.append(row)

   
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    
    if rows:
        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("No pairs found.\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare all ImageJ vs Our PSF TIFF pairs per model/channel.")
    ap.add_argument(
        "--generated_dir",
        type=str,
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\generated",
        help="Folder containing both ImageJ PSFs and generated PSFs.",
    )
    ap.add_argument(
        "--out_txt",
        type=str,
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\psf_compare_report_ALL.txt",
        help="Output text report path.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\psf\psf_compare_report_ALL.csv",
        help="Output CSV summary path.",
    )
    args = ap.parse_args()

    generated_dir = Path(args.generated_dir)
    if not generated_dir.exists():
        raise FileNotFoundError(f"generated_dir does not exist: {generated_dir}")

    pairs = discover_pairs(generated_dir)

    # also show missing expected pairs (optional: quick sanity list)

    write_reports(pairs, Path(args.out_txt), Path(args.out_csv))
    print(f"Done. Pairs compared: {len(pairs)}")
    print(f"Report TXT: {args.out_txt}")
    print(f"Report CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())