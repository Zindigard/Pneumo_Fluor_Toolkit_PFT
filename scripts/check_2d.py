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
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

"""Checks 2D image files and shows their basic content"""

try:
    import zarr
except Exception as e:
    raise ImportError(
        "This script needs zarr. Install optional deps:\n"
        "  pip install zarr ome-zarr numcodecs\n"
    ) from e


REPO_ROOT = find_repo_root(_THIS_FILE)
IN_2D_TIME = REPO_ROOT / "results" / "img" / "2d_time"
IN_2D_WGA = REPO_ROOT / "results" / "img" / "2d_wga_dapi"
OUT_ROOT = REPO_ROOT / "results" / "2d"


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v).strip()
    return s if s else "-"


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def robust_noise_from_differences(img2d: np.ndarray) -> float:
    """
    Robust noise sigma estimate from adjacent-pixel differences.
    sigma ~= MAD(diff)/sqrt(2)
    """
    s = np.asarray(img2d)
    s = s.astype(np.float32, copy=False)
    if s.ndim != 2:
        s = np.squeeze(s)
    if s.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img2d.shape}")

    dx = s[:, 1:] - s[:, :-1]
    dy = s[1:, :] - s[:-1, :]

    diffs = np.concatenate([dx.ravel(), dy.ravel()])
    sig_diff = _mad_sigma(diffs)
    return float(sig_diff / np.sqrt(2))


def _read_omezarr_axes_and_array(zarr_path: Path):
    root = zarr.open_group(str(zarr_path), mode="r")
    ms = root.attrs.get("multiscales", None)
    if not ms:
        raise ValueError(f"No multiscales in {zarr_path}")
    ms0 = ms[0]
    axes_list = ms0.get("axes", [])
    axes = [a["name"] if isinstance(a, dict) else str(a) for a in axes_list]
    rel = ms0["datasets"][0]["path"]
    arr = root[rel]
    return axes, arr


@dataclass
class Noise2DResult:
    dataset: str              # "2d_time" or "2d_wga_dapi"
    sample_rel: str
    zarr_path: Path
    axes: str
    shape: tuple[int, ...]
    noise_summary: dict[str, float]   # e.g. {"overall":..., "c0":..., "c1":..., "t0":...}
    out_txt: Path


def _relative_to_any(p: Path, base: Path) -> str:
    try:
        return p.relative_to(base).as_posix()
    except Exception:
        return p.as_posix()


def analyze_2d_time(zarr_path: Path) -> Noise2DResult:
    axes, arr = _read_omezarr_axes_and_array(zarr_path)
    axes_str = "".join(axes)
    shape = tuple(arr.shape)

    # Default pick first channel if 'c' exists
    ic = axes.index("c") if "c" in axes else None
    it = axes.index("t") if "t" in axes else None
    iy = axes.index("y")
    ix = axes.index("x")

    def slice_at(t: int | None):
        # Fix all non-yx dims to 0, vary t if exists
        idx: list[Any] = []
        for a in axes:
            if a == "y":
                idx.append(slice(None))
            elif a == "x":
                idx.append(slice(None))
            elif a == "t":
                idx.append(t if t is not None else 0)
            elif a == "c":
                idx.append(0)  # channel 0 for time dataset
            else:
                idx.append(0)
        im = np.asarray(arr[tuple(idx)])
        im = np.squeeze(im)
        return im

    noise_summary: dict[str, float] = {}

    if it is None:
        # single frame
        n = robust_noise_from_differences(slice_at(None))
        noise_summary["overall"] = n
    else:
        T = int(shape[it])
        noises = np.zeros(T, dtype=np.float64)
        for t in range(T):
            noises[t] = robust_noise_from_differences(slice_at(t))
            noise_summary[f"t{t}"] = float(noises[t])
        noise_summary["overall"] = float(np.mean(noises))
        noise_summary["min_t"] = float(np.min(noises))
        noise_summary["max_t"] = float(np.max(noises))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_rel = _relative_to_any(zarr_path, IN_2D_TIME)
    safe_name = sample_rel.replace("/", "__").replace(":", "")
    out_txt = OUT_ROOT / f"{safe_name}__noise2d.txt"

    lines = []
    lines += ["# PFT 2D TIME NOISE ANALYSIS", ""]
    lines += [f"OME-Zarr: {zarr_path}"]
    lines += [f"Relative: {sample_rel}"]
    lines += [f"Axes: {axes_str}"]
    lines += [f"Shape: {shape}", ""]
    lines += ["## Noise (robust sigma from pixel differences)"]
    for k in sorted(noise_summary.keys()):
        lines += [f"{k}: {_fmt(noise_summary[k])}"]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return Noise2DResult(
        dataset="2d_time",
        sample_rel=sample_rel,
        zarr_path=zarr_path,
        axes=axes_str,
        shape=shape,
        noise_summary=noise_summary,
        out_txt=out_txt,
    )


def analyze_2d_wga_dapi(zarr_path: Path) -> Noise2DResult:
    axes, arr = _read_omezarr_axes_and_array(zarr_path)
    axes_str = "".join(axes)
    shape = tuple(arr.shape)

    if "y" not in axes or "x" not in axes:
        raise ValueError(f"Missing y/x axes in {zarr_path}: axes={axes}")

    ic = axes.index("c") if "c" in axes else None
    iy = axes.index("y")
    ix = axes.index("x")

    def slice_channel(c: int | None):
        idx: list[Any] = []
        for a in axes:
            if a == "y":
                idx.append(slice(None))
            elif a == "x":
                idx.append(slice(None))
            elif a == "c":
                idx.append(c if c is not None else 0)
            else:
                idx.append(0)
        im = np.asarray(arr[tuple(idx)])
        im = np.squeeze(im)
        return im

    noise_summary: dict[str, float] = {}

    if ic is None:
        n = robust_noise_from_differences(slice_channel(None))
        noise_summary["overall"] = n
    else:
        C = int(shape[ic])
        noises = []
        for c in range(C):
            nc = robust_noise_from_differences(slice_channel(c))
            noise_summary[f"c{c}"] = float(nc)
            noises.append(nc)
        noise_summary["overall"] = float(np.mean(noises))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_rel = _relative_to_any(zarr_path, IN_2D_WGA)
    safe_name = sample_rel.replace("/", "__").replace(":", "")
    out_txt = OUT_ROOT / f"{safe_name}__noise2d.txt"

    lines = []
    lines += ["# PFT 2D TWO-CHANNEL NOISE ANALYSIS", ""]
    lines += [f"OME-Zarr: {zarr_path}"]
    lines += [f"Relative: {sample_rel}"]
    lines += [f"Axes: {axes_str}"]
    lines += [f"Shape: {shape}", ""]
    lines += ["## Noise (robust sigma from pixel differences)"]
    for k in sorted(noise_summary.keys()):
        lines += [f"{k}: {_fmt(noise_summary[k])}"]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return Noise2DResult(
        dataset="2d_wga_dapi",
        sample_rel=sample_rel,
        zarr_path=zarr_path,
        axes=axes_str,
        shape=shape,
        noise_summary=noise_summary,
        out_txt=out_txt,
    )


def find_all_omezarr(base: Path) -> list[Path]:
    return sorted(base.rglob("image.ome.zarr"))


def write_summary_csv(results: list[Noise2DResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect all possible keys
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.noise_summary.keys())

    keys_sorted = sorted(all_keys)

    fields = ["dataset", "sample_rel", "zarr_path", "axes", "shape"] + keys_sorted + ["out_txt"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {
                "dataset": r.dataset,
                "sample_rel": r.sample_rel,
                "zarr_path": str(r.zarr_path),
                "axes": r.axes,
                "shape": str(r.shape),
                "out_txt": str(r.out_txt),
            }
            for k in keys_sorted:
                row[k] = r.noise_summary.get(k, None)
            w.writerow(row)


def write_all_statistics_txt(results: list[Noise2DResult], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        out_txt.write_text("No 2D results.\n", encoding="utf-8")
        return

    # Dataset split
    time_res = [r for r in results if r.dataset == "2d_time"]
    wga_res = [r for r in results if r.dataset == "2d_wga_dapi"]

    def stats_for(key: str, rs: list[Noise2DResult]) -> str:
        vals = []
        for r in rs:
            v = r.noise_summary.get(key, None)
            if v is None:
                continue
            vals.append(float(v))
        a = np.array(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return f"{key}: -"
        return f"{key}: min={a.min():.6g}, median={np.median(a):.6g}, mean={a.mean():.6g}, max={a.max():.6g}"

    lines = []
    lines += ["# PFT ALL 2D NOISE STATISTICS", ""]
    lines += [f"Total files: {len(results)}"]
    lines += [f"2d_time files: {len(time_res)}"]
    lines += [f"2d_wga_dapi files: {len(wga_res)}", ""]

    lines += ["## 2D TIME (overall noise)"]
    lines += [stats_for("overall", time_res)]
    lines += [stats_for("min_t", time_res)]
    lines += [stats_for("max_t", time_res)]
    lines += [""]

    lines += ["## 2D WGA/DAPI (per-channel noise, if present)"]
    lines += [stats_for("overall", wga_res)]
    lines += [stats_for("c0", wga_res)]
    lines += [stats_for("c1", wga_res)]
    lines += [""]

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    z_time = find_all_omezarr(IN_2D_TIME)
    z_wga = find_all_omezarr(IN_2D_WGA)

    if not z_time and not z_wga:
        print("No 2D OME-Zarr files found.")
        print("Expected under:")
        print(" ", IN_2D_TIME)
        print(" ", IN_2D_WGA)
        return

    results: list[Noise2DResult] = []
    errors_txt = OUT_ROOT / "errors_2d_noise.txt"
    if errors_txt.exists():
        errors_txt.unlink()

    for zp in z_time:
        try:
            results.append(analyze_2d_time(zp))
        except Exception as e:
            with errors_txt.open("a", encoding="utf-8") as f:
                f.write(f"{zp}\n  {type(e).__name__}: {e}\n\n")
            print(f"ERROR time: {zp} -> {e}")

    for zp in z_wga:
        try:
            results.append(analyze_2d_wga_dapi(zp))
        except Exception as e:
            with errors_txt.open("a", encoding="utf-8") as f:
                f.write(f"{zp}\n  {type(e).__name__}: {e}\n\n")
            print(f"ERROR wga/dapi: {zp} -> {e}")

    # Outputs
    out_csv = OUT_ROOT / "summary_2d_noise.csv"
    write_summary_csv(results, out_csv)

    out_all = OUT_ROOT / "ALL_2D_NOISE_STATISTICS.txt"
    write_all_statistics_txt(results, out_all)

    print("\nDone.")
    print("Per-sample TXT + summary written to:", OUT_ROOT)
    print("CSV:", out_csv)
    print("ALL TXT:", out_all)
    if errors_txt.exists():
        print("Errors:", errors_txt)


if __name__ == "__main__":
    main()
