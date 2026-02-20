from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import tifffile as tiff
except ImportError as e:
    raise ImportError("Missing dependency: tifffile. Install with: pip install tifffile") from e

""" Helper functions for PSF checks and comparisons """


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_tiff_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "name": path.name, "size_bytes": path.stat().st_size}

    with tiff.TiffFile(str(path)) as tf:
        info["n_pages"] = len(tf.pages)
        info["n_series"] = len(tf.series)

        arr = tf.asarray()
        info["array_shape"] = tuple(arr.shape)
        info["dtype"] = str(arr.dtype)

        arr_f = arr.astype(np.float64, copy=False)
        info["min"] = float(np.min(arr_f))
        info["max"] = float(np.max(arr_f))
        info["mean"] = float(np.mean(arr_f))
        info["std"] = float(np.std(arr_f))

        page0 = tf.pages[0]
        tags = {t.name: t.value for t in page0.tags.values()}

        desc = tags.get("ImageDescription", None)
        if isinstance(desc, bytes):
            desc = desc.decode("utf-8", errors="ignore")
        info["has_image_description"] = bool(desc)
        info["image_description_head"] = (desc[:800] if isinstance(desc, str) else str(desc)[:800]) if desc else ""

        info["XResolution"] = tags.get("XResolution", None)
        info["YResolution"] = tags.get("YResolution", None)
        info["ResolutionUnit"] = tags.get("ResolutionUnit", None)

        spacing = None
        if isinstance(desc, str):
            for line in desc.splitlines():
                if line.strip().startswith("spacing="):
                    try:
                        spacing = float(line.strip().split("=", 1)[1])
                    except Exception:
                        spacing = None
                    break
        info["imagej_spacing"] = spacing

    info["md5"] = md5_file(path)
    return info


def compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"same_shape": a.shape == b.shape, "same_dtype": a.dtype == b.dtype}

    if a.shape != b.shape:
        out["pixel_compare"] = "SKIPPED (shape mismatch)"
        return out

    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    diff = af - bf

    out["max_abs_diff"] = float(np.max(np.abs(diff)))
    out["mae"] = float(np.mean(np.abs(diff)))
    out["mse"] = float(np.mean(diff**2))

    a_std = float(np.std(af))
    b_std = float(np.std(bf))
    out["pearson_corr"] = float(np.corrcoef(af.ravel(), bf.ravel())[0, 1]) if (a_std > 0 and b_std > 0) else None

    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        out["n_pixels_different_exact"] = int(np.count_nonzero(a != b))

    return out


def find_project_root(start: Path) -> Path:
    markers = ["pyproject.toml", "setup.cfg", "src", ".git"]
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return start.resolve().parent.parent


def _resolve_psf_path(project_root: Path, p: str) -> Path:
    """Accept either absolute path, or filename relative to <project_root>/results/psf."""
    path = Path(p)
    if path.exists():
        return path
    candidate = project_root / "results" / "psf" / "generated" / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve PSF path: '{p}'. Tried '{path}' and '{candidate}'.")


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)

    default_a = "PSF BW Blue.tif"
    default_b = "psf_BW_TV1-T1-SR_Lambda405nm.tif"

    ap = argparse.ArgumentParser(
        description=(
            "Compare two PSF TIFFs (metadata + per-pixel stats if shapes match).\n"
            "Defaults: BW Blue (ImageJ) vs generated BW TV1-T1-SR Lambda405nm."
        )
    )
    ap.add_argument("--a", default=default_a, help=f"First PSF tif (abs path or filename in results/psf). Default: {default_a}")
    ap.add_argument("--b", default=default_b, help=f"Second PSF tif (abs path or filename in results/psf). Default: {default_b}")
    ap.add_argument("--out", default=None, help="Output report txt path (default: results/psf/psf_compare_report_custom.txt)")
    args = ap.parse_args()

    a_path = _resolve_psf_path(project_root, args.a)
    b_path = _resolve_psf_path(project_root, args.b)

    a_info = read_tiff_info(a_path)
    b_info = read_tiff_info(b_path)

    a_arr = tiff.imread(str(a_path))
    b_arr = tiff.imread(str(b_path))
    comp = compare_arrays(a_arr, b_arr)

    lines = []
    lines.append("PSF TIFF COMPARISON REPORT (CUSTOM)")
    lines.append("=" * 60)
    lines.append(f"Project root: {project_root}")
    lines.append("")
    lines.append("FILES")
    lines.append("-" * 60)
    lines.append(f"A: {a_path}")
    lines.append(f"B: {b_path}")
    lines.append("")

    def dump_info(title: str, info: Dict[str, Any]) -> None:
        lines.append(title)
        lines.append("-" * 60)
        lines.append(f"Path: {info['path']}")
        lines.append(f"Size (bytes): {info['size_bytes']}")
        lines.append(f"MD5: {info['md5']}")
        lines.append(f"Pages: {info['n_pages']} | Series: {info['n_series']}")
        lines.append(f"Array shape: {info['array_shape']}")
        lines.append(f"Dtype: {info['dtype']}")
        lines.append(f"Stats: min={info['min']:.6g} max={info['max']:.6g} mean={info['mean']:.6g} std={info['std']:.6g}")
        lines.append(f"XResolution tag: {info['XResolution']}")
        lines.append(f"YResolution tag: {info['YResolution']}")
        lines.append(f"ResolutionUnit tag: {info['ResolutionUnit']}")
        lines.append(f"ImageJ spacing (from ImageDescription): {info['imagej_spacing']}")
        lines.append(f"Has ImageDescription: {info['has_image_description']}")
        if info["image_description_head"]:
            lines.append("ImageDescription (first ~800 chars):")
            lines.append(info["image_description_head"])
        lines.append("")

    dump_info("A FILE INFO", a_info)
    dump_info("B FILE INFO", b_info)

    lines.append("COMPARISON")
    lines.append("-" * 60)
    lines.append(f"Same shape: {comp.get('same_shape')}")
    lines.append(f"Same dtype: {comp.get('same_dtype')}")
    if comp.get("pixel_compare") == "SKIPPED (shape mismatch)":
        lines.append("Per-pixel comparison skipped due to shape mismatch.")
    else:
        lines.append(f"Max abs diff: {comp.get('max_abs_diff')}")
        lines.append(f"MAE: {comp.get('mae')}")
        lines.append(f"MSE: {comp.get('mse')}")
        lines.append(f"Pearson corr: {comp.get('pearson_corr')}")
        if "n_pixels_different_exact" in comp:
            lines.append(f"Pixels different (exact): {comp['n_pixels_different_exact']}")
    lines.append("")

    out_path = Path(args.out) if args.out else (project_root / "results" / "psf" / "psf_compare_report_custom.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())