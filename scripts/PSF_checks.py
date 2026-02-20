from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import tifffile as tiff
except ImportError as e:
    raise ImportError("Missing dependency: tifffile. Install with: pip install tifffile") from e


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


def choose_files(psf_dir: Path) -> Tuple[Path, Path]:
    theo = psf_dir / "theoretical_psf.tif"
    if not theo.exists():
        raise FileNotFoundError(f"Missing: {theo}")

    ij = psf_dir / "PSF BW.tif"
    if ij.exists():
        return theo, ij

    candidates = sorted(psf_dir.glob("PSF*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Missing ImageJ PSF file. Expected 'PSF BW.tif' or any 'PSF*.tif' in {psf_dir}"
        )
    return theo, candidates[0]


def find_project_root(start: Path) -> Path:
    # Walk up until we find a folder that looks like your repo root
    # (adjust markers if needed)
    markers = ["pyproject.toml", "setup.cfg", "src", ".git"]
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    # fallback: scripts/.. = repo root
    return start.resolve().parent.parent


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)

    psf_dir = project_root / "results" / "psf"
    psf_dir.mkdir(parents=True, exist_ok=True)

    theo_path, ij_path = choose_files(psf_dir)

    theo_info = read_tiff_info(theo_path)
    ij_info = read_tiff_info(ij_path)

    theo_arr = tiff.imread(str(theo_path))
    ij_arr = tiff.imread(str(ij_path))
    comp = compare_arrays(theo_arr, ij_arr)

    lines = []
    lines.append("PSF TIFF COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append(f"Project root: {project_root}")
    lines.append(f"PSF folder : {psf_dir}")
    lines.append("")
    lines.append("FILES")
    lines.append("-" * 60)
    lines.append(f"1) Our PSF   : {theo_path.name}")
    lines.append(f"2) ImageJ PSF: {ij_path.name}")
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

    dump_info("OUR PSF FILE INFO", theo_info)
    dump_info("IMAGEJ PSF FILE INFO", ij_info)

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

    report_path = psf_dir / "psf_compare_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())