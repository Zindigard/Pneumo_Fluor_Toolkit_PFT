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
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np

"""Checks 3D image files and helps confirm the volume data looks correct before further processing"""

try:
    import zarr
except Exception as e:
    raise ImportError(
        "This script needs zarr. Install optional deps:\n"
        "  pip install zarr ome-zarr numcodecs\n"
    ) from e


REPO_ROOT = find_repo_root(_THIS_FILE)
IN_ROOT = REPO_ROOT / "results" / "img" / "3d_data"
OUT_ROOT = REPO_ROOT / "results" / "3d"



def _safe_float(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).strip() if str(v).strip() else "-"


def _mad_sigma(x: np.ndarray) -> float:
    """Robust sigma via MAD: sigma ~= 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _robust_noise_from_differences(slice2d: np.ndarray) -> float:
    """
    Noise estimate from adjacent-pixel differences using robust MAD.
    For i.i.d. noise: std(diff) = sqrt(2)*sigma => sigma ≈ std(diff)/sqrt(2).
    """
    s = np.asarray(slice2d)
    if s.ndim != 2:
        raise ValueError("slice2d must be 2D")
    s = s.astype(np.float32, copy=False)

    dx = s[:, 1:] - s[:, :-1]
    dy = s[1:, :] - s[:-1, :]

    diffs = np.concatenate([dx.ravel(), dy.ravel()])
    sig_diff = _mad_sigma(diffs)
    return float(sig_diff / np.sqrt(2))


def _linear_fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope b of y ~ a + b*x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    xm = x.mean()
    ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum((x - xm) * (y - ym)) / denom)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    xs = x - x.mean()
    ys = y - y.mean()
    denom = np.sqrt(np.sum(xs**2) * np.sum(ys**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(xs * ys) / denom)


def _close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _stats_line(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{name}: -"
    return (
        f"{name}: min={arr.min():.6g}, p25={np.percentile(arr,25):.6g}, "
        f"median={np.median(arr):.6g}, mean={arr.mean():.6g}, "
        f"p75={np.percentile(arr,75):.6g}, max={arr.max():.6g}"
    )


def _noise_class_from_percentiles(x: float, p33: float, p66: float) -> str:
    """Low/medium/high based on dataset percentiles (adaptive)."""
    if not np.isfinite(x):
        return "unknown"
    if x <= p33:
        return "low"
    if x <= p66:
        return "medium"
    return "high"


@dataclass(frozen=True)
class OmezarrInfo:
    zarr_path: Path
    dataset_rel: str
    axes: list[str]
    shape: tuple[int, ...]
    scale: dict[str, float]     # axis -> scale (µm for z/y/x if present)
    zyx_size: tuple[int, int, int]  # Z,Y,X
    has_channel: bool
    channel_count: int | None


def _read_omezarr_info(zarr_path: Path) -> OmezarrInfo:
    root = zarr.open_group(str(zarr_path), mode="r")

    ms = root.attrs.get("multiscales", None)
    if not ms:
        raise ValueError(f"No multiscales found in: {zarr_path}")

    ms0 = ms[0]
    axes_list = ms0.get("axes", [])
    axes = [a["name"] if isinstance(a, dict) else str(a) for a in axes_list]

    datasets = ms0.get("datasets", [])
    if not datasets:
        raise ValueError(f"No datasets in multiscales for: {zarr_path}")

    rel = datasets[0]["path"]
    arr = root[rel]
    shape = tuple(arr.shape)

    scale_vec: list[float] | None = None
    ct = datasets[0].get("coordinateTransformations", None)
    if ct and isinstance(ct, list):
        for t in ct:
            if isinstance(t, dict) and t.get("type") == "scale":
                scale_vec = t.get("scale")
                break

    scale_map: dict[str, float] = {}
    if scale_vec is not None and len(scale_vec) == len(axes):
        for a, s in zip(axes, scale_vec):
            try:
                scale_map[a] = float(s)
            except Exception:
                pass

    if "z" not in axes or "y" not in axes or "x" not in axes:
        raise ValueError(f"Axes missing z/y/x in {zarr_path}: axes={axes}")

    iz = axes.index("z")
    iy = axes.index("y")
    ix = axes.index("x")
    zyx = (shape[iz], shape[iy], shape[ix])

    has_c = "c" in axes
    c_count = shape[axes.index("c")] if has_c else None

    try:
        dataset_rel = zarr_path.relative_to(IN_ROOT).as_posix()
    except Exception:
        dataset_rel = zarr_path.as_posix()

    return OmezarrInfo(
        zarr_path=zarr_path,
        dataset_rel=dataset_rel,
        axes=axes,
        shape=shape,
        scale=scale_map,
        zyx_size=(int(zyx[0]), int(zyx[1]), int(zyx[2])),
        has_channel=has_c,
        channel_count=int(c_count) if c_count is not None else None,
    )


@dataclass(frozen=True)
class ReportMeta:
    voxel_dx_um: float | None
    voxel_dy_um: float | None
    voxel_dz_um: float | None
    axes_header: str | None


_RE_VOX = re.compile(
    r"Voxel size\s*\(µm\)\s*:\s*dx=(?P<dx>[-+0-9.eE]+|[-])\s*,\s*dy=(?P<dy>[-+0-9.eE]+|[-])\s*,\s*dz=(?P<dz>[-+0-9.eE]+|[-])"
)
_RE_AXES = re.compile(r"CZI axes\s*\(header\)\s*:\s*(?P<axes>.+)$")


def _parse_report_txt(path: Path) -> ReportMeta:
    if not path.exists():
        return ReportMeta(None, None, None, None)

    txt = path.read_text(encoding="utf-8", errors="ignore")

    dx = dy = dz = None
    m = _RE_VOX.search(txt)
    if m:
        dx = _safe_float(m.group("dx")) if m.group("dx") != "-" else None
        dy = _safe_float(m.group("dy")) if m.group("dy") != "-" else None
        dz = _safe_float(m.group("dz")) if m.group("dz") != "-" else None

    axes_header = None
    m2 = _RE_AXES.search(txt)
    if m2:
        axes_header = m2.group("axes").strip()
        if axes_header == "-" or axes_header.lower() == "none":
            axes_header = None

    return ReportMeta(dx, dy, dz, axes_header)


def _match_str(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "unknown"
    return "match" if _close(a, b, tol=1e-6) else "DIFF"



META_PATTERNS: dict[str, re.Pattern] = {
    "modality": re.compile(r"^Modality:\s*(?P<v>.*)$", re.MULTILINE),
    "sim_type": re.compile(r"^SIM type:\s*(?P<v>.*)$", re.MULTILINE),
    "recon_software": re.compile(r"^Reconstruction software:\s*(?P<v>.*)$", re.MULTILINE),
    "recon_version": re.compile(r"^Reconstruction version:\s*(?P<v>.*)$", re.MULTILINE),
    "exported_image_type": re.compile(r"^Exported image type.*:\s*(?P<v>.*)$", re.MULTILINE),
    "microscope_system": re.compile(r"^Microscope / system:\s*(?P<v>.*)$", re.MULTILINE),
    "objective_model": re.compile(r"^Objective model:\s*(?P<v>.*)$", re.MULTILINE),
    "objective_mag": re.compile(r"^Objective magnification:\s*(?P<v>.*)$", re.MULTILINE),
    "objective_na": re.compile(r"^Objective NA:\s*(?P<v>.*)$", re.MULTILINE),
    "immersion": re.compile(r"^Immersion medium:\s*(?P<v>.*)$", re.MULTILINE),
    "ri_immersion": re.compile(r"^Refractive index \(immersion\):\s*(?P<v>.*)$", re.MULTILINE),
    "ri_sample": re.compile(r"^Refractive index \(sample/mount\):\s*(?P<v>.*)$", re.MULTILINE),
    "detector_type": re.compile(r"^Detector type:\s*(?P<v>.*)$", re.MULTILINE),
    "detector_model": re.compile(r"^Detector model:\s*(?P<v>.*)$", re.MULTILINE),
    "voxel_um": _RE_VOX,
}


def parse_metadata_report_fields(report_path: Path) -> dict[str, str | None]:
    """
    Returns dict(field -> value).
    Missing or '-' becomes None.
    """
    if not report_path.exists():
        return {}

    txt = report_path.read_text(encoding="utf-8", errors="ignore")

    out: dict[str, str | None] = {}
    for k, pat in META_PATTERNS.items():
        m = pat.search(txt)
        if not m:
            out[k] = None
            continue

        if k == "voxel_um":
            dx = m.group("dx")
            dy = m.group("dy")
            dz = m.group("dz")
            if dx == "-" or dy == "-" or dz == "-":
                out[k] = None
            else:
                out[k] = f"dx={dx}, dy={dy}, dz={dz}"
        else:
            v = m.group("v").strip()
            if v == "-" or v == "":
                out[k] = None
            else:
                out[k] = v

    return out


def write_metadata_uniformity_txt(
    dataset_to_fields: dict[str, dict[str, str | None]],
    out_txt: Path,
) -> None:
    """
    For each field:
      - if identical across ALL available files -> show that value
      - else -> show all distinct values + counts, plus missing count
    """
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    all_fields: set[str] = set()
    for d in dataset_to_fields.values():
        all_fields.update(d.keys())

    field_order = [
        "modality",
        "sim_type",
        "recon_software",
        "recon_version",
        "exported_image_type",
        "microscope_system",
        "objective_model",
        "objective_mag",
        "objective_na",
        "immersion",
        "ri_immersion",
        "ri_sample",
        "detector_type",
        "detector_model",
        "voxel_um",
    ]
    for f in sorted(all_fields):
        if f not in field_order:
            field_order.append(f)

    lines: list[str] = []
    lines += ["# PFT 3D METADATA UNIFORMITY AUDIT", ""]
    lines += [f"Files with reports: {len(dataset_to_fields)}", ""]

    for field in field_order:
        # gather values
        vals: list[str | None] = []
        for ds, d in dataset_to_fields.items():
            vals.append(d.get(field, None))

        missing = sum(v is None for v in vals)
        present = [v for v in vals if v is not None]

        lines += [f"## Field: {field}"]
        if not present:
            lines += [f"All missing ({missing}/{len(vals)}).", ""]
            continue

        uniq = {}
        for v in present:
            uniq[v] = uniq.get(v, 0) + 1

        if len(uniq) == 1 and missing == 0:
            only_val = next(iter(uniq.keys()))
            lines += [f"Identical across all files: {only_val}", ""]
        else:
            lines += [f"Not identical. Missing: {missing}/{len(vals)}"]
            lines += ["Distinct values (count):"]
            for v, c in sorted(uniq.items(), key=lambda kv: (-kv[1], str(kv[0]))):
                lines += [f"  - {v}  ({c})"]
            lines += [""]

    # optional: list which files have missing critical fields
    critical = ["objective_na", "objective_model", "voxel_um", "detector_model", "recon_software"]
    lines += ["## Missing critical fields (per file)"]
    for ds, d in dataset_to_fields.items():
        miss = [k for k in critical if d.get(k) in (None, "")]
        if miss:
            lines += [f"{ds}: missing {', '.join(miss)}"]
    lines += [""]

    out_txt.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class AnalysisResult:
    dataset_rel: str
    zarr_path: Path
    volume_zyx: tuple[int, int, int]
    axes: str

    scale_dx: float | None
    scale_dy: float | None
    scale_dz: float | None

    report_dx: float | None
    report_dy: float | None
    report_dz: float | None

    voxel_match_dx: str
    voxel_match_dy: str
    voxel_match_dz: str

    # Noise summary
    noise_mean: float
    noise_std: float
    noise_cv: float
    noise_p10: float
    noise_p90: float

    noise_slope_per_slice: float
    noise_corr_with_depth: float

    # Mean intensity (bleaching proxy)
    intensity_mean_slope_per_slice: float
    intensity_corr_with_depth: float

    # Classification (filled later)
    noise_class: str | None = None

    out_txt: Path | None = None


def analyze_one(zarr_path: Path) -> AnalysisResult:
    info = _read_omezarr_info(zarr_path)

    sample_dir = zarr_path.parent  # .../<file_stem>/
    report_path = sample_dir / "metadata_report.txt"
    report = _parse_report_txt(report_path)

    root = zarr.open_group(str(zarr_path), mode="r")
    rel = root.attrs["multiscales"][0]["datasets"][0]["path"]
    arr = root[rel]

    axes = info.axes
    iz, iy, ix = axes.index("z"), axes.index("y"), axes.index("x")

    Z = info.zyx_size[0]

    noise_per_z = np.zeros(Z, dtype=np.float64)
    mean_per_z = np.zeros(Z, dtype=np.float64)

   
    base_idx: list[Any] = []
    for a in axes:
        if a == "z":
            base_idx.append(slice(None))  # overwritten per z
        elif a in ("y", "x"):
            base_idx.append(slice(None))
        elif a == "c":
            base_idx.append(0)
        else:
            base_idx.append(0)

    for z in range(Z):
        idx = list(base_idx)
        idx[iz] = z
        sl = np.asarray(arr[tuple(idx)], dtype=np.float32)
        sl = np.squeeze(sl)
        if sl.ndim != 2:
            raise ValueError(f"Unexpected slice ndim={sl.ndim} for {zarr_path}, axes={axes}")

        noise_per_z[z] = _robust_noise_from_differences(sl)
        mean_per_z[z] = float(np.mean(sl))

    depth = np.arange(Z, dtype=np.float64)

    noise_slope = _linear_fit_slope(depth, noise_per_z)
    noise_corr = _corr(depth, noise_per_z)

    mean_slope = _linear_fit_slope(depth, mean_per_z)
    mean_corr = _corr(depth, mean_per_z)

    dx_zarr = info.scale.get("x")
    dy_zarr = info.scale.get("y")
    dz_zarr = info.scale.get("z")

    
    noise_mean = float(np.mean(noise_per_z))
    noise_std = float(np.std(noise_per_z))
    noise_cv = float(noise_std / (noise_mean + 1e-12))
    noise_p10 = float(np.percentile(noise_per_z, 10))
    noise_p90 = float(np.percentile(noise_per_z, 90))

    # Save per-sample TXT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = info.dataset_rel.replace("/", "__").replace(":", "")
    out_txt = OUT_ROOT / f"{safe_name}__volume_noise.txt"

    lines: list[str] = []
    lines += ["# PFT 3D OME-Zarr analysis", ""]
    lines += [f"OME-Zarr: {zarr_path}"]
    lines += [f"Relative: {info.dataset_rel}", ""]
    lines += ["## Volume size (from OME-Zarr)"]
    lines += [f"Axes: {''.join(info.axes)}"]
    lines += [f"Shape: {info.shape}"]
    lines += [f"Volume Z×Y×X: {info.zyx_size[0]} × {info.zyx_size[1]} × {info.zyx_size[2]}"]
    lines += [""]

    lines += ["## Voxel size consistency (report vs OME-Zarr scale)"]
    lines += [f"Zarr scale (µm): dx={_fmt(dx_zarr)}, dy={_fmt(dy_zarr)}, dz={_fmt(dz_zarr)}"]
    lines += [f"Report voxel (µm): dx={_fmt(report.voxel_dx_um)}, dy={_fmt(report.voxel_dy_um)}, dz={_fmt(report.voxel_dz_um)}"]
    lines += [f"Match: dx={_match_str(report.voxel_dx_um, dx_zarr)}, dy={_match_str(report.voxel_dy_um, dy_zarr)}, dz={_match_str(report.voxel_dz_um, dz_zarr)}"]
    if report.axes_header is not None:
        lines += [f"Report CZI axes(header): {report.axes_header}"]
    lines += [""]

    lines += ["## Noise across Z (robust estimate per slice)"]
    lines += [f"Noise mean: {_fmt(noise_mean)}"]
    lines += [f"Noise std: {_fmt(noise_std)}"]
    lines += [f"Noise CV (std/mean): {_fmt(noise_cv)}"]
    lines += [f"Noise p10: {_fmt(noise_p10)}"]
    lines += [f"Noise p90: {_fmt(noise_p90)}"]
    lines += [f"Noise slope per slice: {_fmt(noise_slope)}  (positive => increases with depth)"]
    lines += [f"Noise corr(depth, noise): {_fmt(noise_corr)}"]
    lines += [""]

    lines += ["## Mean intensity across Z (bleaching proxy)"]
    lines += [f"Mean intensity slope per slice: {_fmt(mean_slope)}  (negative => decreases with depth)"]
    lines += [f"Mean intensity corr(depth, mean): {_fmt(mean_corr)}"]
    lines += [""]

    lines += ["## Per-slice table"]
    lines += ["z\tnoise_sigma\tmean_intensity"]
    for z in range(Z):
        lines += [f"{z}\t{noise_per_z[z]:.6g}\t{mean_per_z[z]:.6g}"]

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return AnalysisResult(
        dataset_rel=info.dataset_rel,
        zarr_path=info.zarr_path,
        volume_zyx=info.zyx_size,
        axes="".join(info.axes),
        scale_dx=dx_zarr,
        scale_dy=dy_zarr,
        scale_dz=dz_zarr,
        report_dx=report.voxel_dx_um,
        report_dy=report.voxel_dy_um,
        report_dz=report.voxel_dz_um,
        voxel_match_dx=_match_str(report.voxel_dx_um, dx_zarr),
        voxel_match_dy=_match_str(report.voxel_dy_um, dy_zarr),
        voxel_match_dz=_match_str(report.voxel_dz_um, dz_zarr),
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_cv=noise_cv,
        noise_p10=noise_p10,
        noise_p90=noise_p90,
        noise_slope_per_slice=noise_slope,
        noise_corr_with_depth=noise_corr,
        intensity_mean_slope_per_slice=mean_slope,
        intensity_corr_with_depth=mean_corr,
        noise_class=None, 
        out_txt=out_txt,
    )



def find_all_omezarr(in_root: Path) -> list[Path]:
    return sorted(in_root.rglob("image.ome.zarr"))


def write_summary_csv(results: Iterable[AnalysisResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset_rel",
        "zarr_path",
        "axes",
        "Z",
        "Y",
        "X",
        "scale_dx_um",
        "scale_dy_um",
        "scale_dz_um",
        "report_dx_um",
        "report_dy_um",
        "report_dz_um",
        "match_dx",
        "match_dy",
        "match_dz",
        "noise_class",
        "noise_mean",
        "noise_std",
        "noise_cv",
        "noise_p10",
        "noise_p90",
        "noise_slope_per_slice",
        "noise_corr_with_depth",
        "intensity_mean_slope_per_slice",
        "intensity_corr_with_depth",
        "out_txt",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "dataset_rel": r.dataset_rel,
                    "zarr_path": str(r.zarr_path),
                    "axes": r.axes,
                    "Z": r.volume_zyx[0],
                    "Y": r.volume_zyx[1],
                    "X": r.volume_zyx[2],
                    "scale_dx_um": r.scale_dx,
                    "scale_dy_um": r.scale_dy,
                    "scale_dz_um": r.scale_dz,
                    "report_dx_um": r.report_dx,
                    "report_dy_um": r.report_dy,
                    "report_dz_um": r.report_dz,
                    "match_dx": r.voxel_match_dx,
                    "match_dy": r.voxel_match_dy,
                    "match_dz": r.voxel_match_dz,
                    "noise_class": r.noise_class,
                    "noise_mean": r.noise_mean,
                    "noise_std": r.noise_std,
                    "noise_cv": r.noise_cv,
                    "noise_p10": r.noise_p10,
                    "noise_p90": r.noise_p90,
                    "noise_slope_per_slice": r.noise_slope_per_slice,
                    "noise_corr_with_depth": r.noise_corr_with_depth,
                    "intensity_mean_slope_per_slice": r.intensity_mean_slope_per_slice,
                    "intensity_corr_with_depth": r.intensity_corr_with_depth,
                    "out_txt": str(r.out_txt) if r.out_txt else "",
                }
            )


def write_all_files_statistics_txt(
    results: list[AnalysisResult],
    out_txt: Path,
    *,
    corr_strong: float = 0.5,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        out_txt.write_text("No successful results to summarize.\n", encoding="utf-8")
        return

    n = len(results)

    Zs = np.array([r.volume_zyx[0] for r in results], dtype=np.float64)
    Ys = np.array([r.volume_zyx[1] for r in results], dtype=np.float64)
    Xs = np.array([r.volume_zyx[2] for r in results], dtype=np.float64)

    dx_match = sum(r.voxel_match_dx == "match" for r in results)
    dy_match = sum(r.voxel_match_dy == "match" for r in results)
    dz_match = sum(r.voxel_match_dz == "match" for r in results)

    dx_diff = sum(r.voxel_match_dx == "DIFF" for r in results)
    dy_diff = sum(r.voxel_match_dy == "DIFF" for r in results)
    dz_diff = sum(r.voxel_match_dz == "DIFF" for r in results)

    dx_unk = sum(r.voxel_match_dx == "unknown" for r in results)
    dy_unk = sum(r.voxel_match_dy == "unknown" for r in results)
    dz_unk = sum(r.voxel_match_dz == "unknown" for r in results)

    noise_mean = np.array([r.noise_mean for r in results], dtype=np.float64)
    noise_std = np.array([r.noise_std for r in results], dtype=np.float64)
    noise_cv = np.array([r.noise_cv for r in results], dtype=np.float64)
    noise_slope = np.array([r.noise_slope_per_slice for r in results], dtype=np.float64)
    noise_corr = np.array([r.noise_corr_with_depth for r in results], dtype=np.float64)

    mean_slope = np.array([r.intensity_mean_slope_per_slice for r in results], dtype=np.float64)
    mean_corr = np.array([r.intensity_corr_with_depth for r in results], dtype=np.float64)

    # thresholds for noise_class already filled; still report distribution
    cls = [r.noise_class or "unknown" for r in results]
    n_low = sum(c == "low" for c in cls)
    n_med = sum(c == "medium" for c in cls)
    n_high = sum(c == "high" for c in cls)
    n_unk = sum(c == "unknown" for c in cls)

    pos_slope = int(np.sum(np.isfinite(noise_slope) & (noise_slope > 0)))
    neg_slope = int(np.sum(np.isfinite(noise_slope) & (noise_slope < 0)))
    strong_inc = int(np.sum(np.isfinite(noise_corr) & (noise_corr >= corr_strong)))
    strong_dec = int(np.sum(np.isfinite(noise_corr) & (noise_corr <= -corr_strong)))

    idx_noise_desc = np.argsort(-noise_mean)
    idx_noise_asc = np.argsort(noise_mean)
    idx_inc = np.argsort(-noise_corr)

    lines: list[str] = []
    lines += ["# PFT 3D ALL FILES STATISTICS", ""]
    lines += [f"Files summarized: {n}", ""]

    lines += ["## Volume size (Z×Y×X) from OME-Zarr"]
    lines += [_stats_line("Z (slices)", Zs)]
    lines += [_stats_line("Y (pixels)", Ys)]
    lines += [_stats_line("X (pixels)", Xs)]
    lines += [""]

    lines += ["## Voxel size consistency (metadata_report.txt vs OME-Zarr scale)"]
    lines += [f"dx: match={dx_match}, DIFF={dx_diff}, unknown={dx_unk}"]
    lines += [f"dy: match={dy_match}, DIFF={dy_diff}, unknown={dy_unk}"]
    lines += [f"dz: match={dz_match}, DIFF={dz_diff}, unknown={dz_unk}"]
    lines += [""]

    lines += ["## Noise corruption per slice (file-level)"]
    lines += [f"class counts: low={n_low}, medium={n_med}, high={n_high}, unknown={n_unk}"]
    lines += [_stats_line("noise_mean (avg over Z)", noise_mean)]
    lines += [_stats_line("noise_std (across Z)", noise_std)]
    lines += [_stats_line("noise_cv = std/mean (across Z)", noise_cv)]
    lines += [""]

    lines += ["## Noise vs depth trend"]
    lines += [_stats_line("noise_slope_per_slice", noise_slope)]
    lines += [_stats_line("noise_corr(depth, noise)", noise_corr)]
    lines += [f"count slope > 0 : {pos_slope} / {n}"]
    lines += [f"count slope < 0 : {neg_slope} / {n}"]
    lines += [f"count corr >= {corr_strong}: {strong_inc} / {n}  (strong increase)"]
    lines += [f"count corr <= -{corr_strong}: {strong_dec} / {n}  (strong decrease)"]
    lines += [""]

    lines += ["## Mean intensity vs depth (bleaching proxy)"]
    lines += [_stats_line("mean_intensity_slope_per_slice", mean_slope)]
    lines += [_stats_line("mean_intensity_corr(depth, mean)", mean_corr)]
    lines += [""]

    lines += ["## Per-file summary table"]
    lines += [
        "file\tZxYxX\tnoise_class\tnoise_mean\tnoise_cv\tnoise_corr(depth)\tmean_corr(depth)\tvoxel_match(dx,dy,dz)"
    ]
    for r in results:
        lines += [
            f"{r.dataset_rel}\t{r.volume_zyx[0]}x{r.volume_zyx[1]}x{r.volume_zyx[2]}\t"
            f"{r.noise_class}\t{r.noise_mean:.6g}\t{r.noise_cv:.6g}\t{r.noise_corr_with_depth:.6g}\t"
            f"{r.intensity_corr_with_depth:.6g}\t{r.voxel_match_dx},{r.voxel_match_dy},{r.voxel_match_dz}"
        ]

    lines += ["", "## Top 10 highest-noise files (by noise_mean)"]
    for k in range(min(10, n)):
        r = results[int(idx_noise_desc[k])]
        lines += [f"[{k+1}] {r.dataset_rel} | noise_mean={r.noise_mean:.6g} | class={r.noise_class} | noise_cv={r.noise_cv:.6g}"]

    lines += ["", "## Top 10 lowest-noise files (by noise_mean)"]
    for k in range(min(10, n)):
        r = results[int(idx_noise_asc[k])]
        lines += [f"[{k+1}] {r.dataset_rel} | noise_mean={r.noise_mean:.6g} | class={r.noise_class} | noise_cv={r.noise_cv:.6g}"]

    lines += ["", "## Top 10 strongest depth-increase (by corr(depth, noise))"]
    for k in range(min(10, n)):
        r = results[int(idx_inc[k])]
        lines += [f"[{k+1}] {r.dataset_rel} | corr(depth,noise)={r.noise_corr_with_depth:.6g} | slope={r.noise_slope_per_slice:.6g} | class={r.noise_class}"]

    out_txt.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    zarrs = find_all_omezarr(IN_ROOT)
    if not zarrs:
        print(f"ERROR: No image.ome.zarr found under: {IN_ROOT}")
        return

    results: list[AnalysisResult] = []
    errors_txt = OUT_ROOT / "errors.txt"
    if errors_txt.exists():
        errors_txt.unlink()

    #  per-file analysis
    print(f"Found {len(zarrs)} OME-Zarr(s). Starting analysis...")
    for i, zp in enumerate(zarrs, start=1):
        print(f"[{i}/{len(zarrs)}] {zp}")
        try:
            res = analyze_one(zp)
            results.append(res)
        except Exception as e:
            with errors_txt.open("a", encoding="utf-8") as f:
                f.write(f"{zp}\n  {type(e).__name__}: {e}\n\n")
            print(f"  ERROR: {type(e).__name__}: {e}")

    if not results:
        print("No successful analyses.")
        return

   
    noise_mean = np.array([r.noise_mean for r in results], dtype=np.float64)
    finite = noise_mean[np.isfinite(noise_mean)]
    p33 = float(np.percentile(finite, 33))
    p66 = float(np.percentile(finite, 66))

    for r in results:
        r.noise_class = _noise_class_from_percentiles(r.noise_mean, p33, p66)

    out_csv = OUT_ROOT / "summary_volume_noise.csv"
    write_summary_csv(results, out_csv)

    
    write_all_files_statistics_txt(results, OUT_ROOT / "ALL_FILES_STATISTICS.txt")

    
    dataset_to_fields: dict[str, dict[str, str | None]] = {}
    for r in results:
        report_path = r.zarr_path.parent / "metadata_report.txt"
        dataset_to_fields[r.dataset_rel] = parse_metadata_report_fields(report_path)

    write_metadata_uniformity_txt(
        dataset_to_fields,
        OUT_ROOT / "METADATA_UNIFORMITY.txt",
    )

    print("\nDone.")
    print(f"Per-sample TXT files in: {OUT_ROOT}")
    print(f"Summary CSV: {out_csv}")
    print(f"ALL FILES TXT: {OUT_ROOT / 'ALL_FILES_STATISTICS.txt'}")
    print(f"METADATA UNIFORMITY TXT: {OUT_ROOT / 'METADATA_UNIFORMITY.txt'}")
    if errors_txt.exists():
        print(f"Errors: {errors_txt}")


if __name__ == "__main__":
    main()
