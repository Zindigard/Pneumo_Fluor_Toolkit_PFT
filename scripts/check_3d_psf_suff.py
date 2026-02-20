from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from PFT.core_prog_parts.decoder_omezar import (
    extract_ome_zarr_meta_for_compare,
    parse_psfgenerator_config,
    check_config_has_required,
)

DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_DATA_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"
DEFAULT_PSF_CFG_DIR = DEFAULT_PROJECT_ROOT / "results" / "psf"
DEFAULT_CFG_NAMES = ("configBW.txt", "configGL.txt", "configRW.txt")


@dataclass(frozen=True)
class CompareResult:
    field: str
    identical: bool
    values_to_files: dict[str, list[str]]  # str(value) -> list[file]


def _gather_ome_zarr_dirs(root: Path) -> list[Path]:
    if root.is_dir() and root.name.endswith(".ome.zarr"):
        return [root]
    dirs = sorted(root.rglob("image.ome.zarr"))
    # accept any *.ome.zarr dirs too
    dirs += [p for p in root.rglob("*.ome.zarr") if p.is_dir() and p.name != "image.ome.zarr"]
    uniq, seen = [], set()
    for d in dirs:
        s = str(d.resolve())
        if s not in seen:
            uniq.append(d)
            seen.add(s)
    return uniq


def _get_field(meta: dict[str, Any], field: str) -> Any:
    if field == "voxel_x":
        return meta.get("voxel_size_um", {}).get("x")
    if field == "voxel_y":
        return meta.get("voxel_size_um", {}).get("y")
    if field == "voxel_z":
        return meta.get("voxel_size_um", {}).get("z")
    return meta.get(field)


def compare_metas(metas: dict[str, dict[str, Any]], fields: Iterable[str]) -> list[CompareResult]:
    out: list[CompareResult] = []
    for f in fields:
        m: dict[str, list[str]] = {}
        for file_id, meta in metas.items():
            val = _get_field(meta, f)
            sval = "None" if val is None else str(val)
            m.setdefault(sval, []).append(file_id)
        out.append(CompareResult(field=f, identical=(len(m) == 1), values_to_files=m))
    return out


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    s = str(v).strip()
    return s if s else "-"


def compare_config_to_image(cfg: dict[str, str], img_meta: dict[str, Any]) -> list[str]:
    """Lightweight consistency checks (sampling + NA)."""
    lines: list[str] = []

    vox = img_meta.get("voxel_size_um", {}) or {}
    vx = _to_float(vox.get("x"))
    vy = _to_float(vox.get("y"))
    vz = _to_float(vox.get("z"))

    # Config sampling: ResLateral/ResAxial are commonly in nm (PSFGenerator), convert to µm if numeric
    res_lat = _to_float(cfg.get("ResLateral"))
    res_ax = _to_float(cfg.get("ResAxial"))
    res_lat_um = (res_lat / 1000.0) if res_lat is not None else None
    res_ax_um = (res_ax / 1000.0) if res_ax is not None else None

    na_cfg = _to_float(cfg.get("NA"))
    na_img = _to_float(img_meta.get("objective_na"))
    if na_cfg is not None and na_img is not None:
        lines.append(f"NA: cfg={na_cfg} vs img={na_img} -> {'OK' if abs(na_cfg-na_img) < 1e-6 else 'MISMATCH'}")
    else:
        lines.append(f"NA: cfg={_fmt(cfg.get('NA'))} vs img={_fmt(img_meta.get('objective_na'))} (cannot fully compare)")

    lines.append(f"ResLateral (µm): cfg={_fmt(res_lat_um)} vs voxel(x)={_fmt(vx)} | voxel(y)={_fmt(vy)}")
    lines.append(f"ResAxial   (µm): cfg={_fmt(res_ax_um)} vs voxel(z)={_fmt(vz)}")
    lines.append(f"Lambda (cfg, nm): {_fmt(cfg.get('Lambda'))} | channel_names (img): {_fmt(img_meta.get('channel_names'))}")
    return lines


def _default_cfg_paths(cfg_dir: Path) -> list[Path]:
    """Return existing BW/GL/RW config paths from the standard results/psf folder."""
    out: list[Path] = []
    for name in DEFAULT_CFG_NAMES:
        p = cfg_dir / name
        if p.exists():
            out.append(p)
    # fallback: any config*.txt
    if not out and cfg_dir.exists():
        out = sorted(cfg_dir.glob("config*.txt"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "STEP2 report: extract + compare OME-Zarr 3D metadata AND check PSFGenerator configs BW/GL/RW.\n"
            "If you run with no arguments, it uses the default Thesis project paths:\n"
            f"  - 3D root: {DEFAULT_3D_DATA_ROOT}\n"
            f"  - PSF cfg: {DEFAULT_PSF_CFG_DIR}"
        )
    )
    ap.add_argument(
        "--root",
        default=str(DEFAULT_3D_DATA_ROOT),
        help="Root folder containing image.ome.zarr (recursive) OR a single *.ome.zarr directory.",
    )
    ap.add_argument(
        "--cfg_dir",
        default=str(DEFAULT_PSF_CFG_DIR),
        help="Folder that contains configBW/configGL/configRW (default: results/psf).",
    )
    ap.add_argument(
        "--cfg",
        nargs="*",
        default=None,
        help=(
            "Optional explicit config file paths. If omitted, script auto-loads "
            "configBW.txt/configGL.txt/configRW.txt from --cfg_dir."
        ),
    )
    ap.add_argument(
        "--ref",
        default=None,
        help="Reference image.ome.zarr for config comparison. If not given, first found under --root is used.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output TXT report path (default: <root>/step2_metadata_and_config_report.txt)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"3D data root not found: {root}\n"
            f"Edit DEFAULT_3D_DATA_ROOT in this script or pass --root <path>."
        )

    zarr_dirs = _gather_ome_zarr_dirs(root)
    if not zarr_dirs:
        raise FileNotFoundError(f"No image.ome.zarr found under: {root}")

    ref = Path(args.ref) if args.ref else zarr_dirs[0]

    # Determine configs
    cfg_dir = Path(args.cfg_dir)
    if args.cfg is None:
        cfg_paths = _default_cfg_paths(cfg_dir)
    else:
        cfg_paths = [Path(p) for p in args.cfg]

    # Extract metas for all
    metas: dict[str, dict[str, Any]] = {}
    for z in zarr_dirs:
        metas[str(z)] = extract_ome_zarr_meta_for_compare(z)

    # Compare across all images
    fields = [
        "axes",
        "shape",
        "dtype",
        "chunks",
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "channel_names",
        "objective_na",
        "objective_magnification",
        "immersion",
        "refractive_index_immersion",
        "refractive_index_sample",
        "modality",
        "sim_mode",
    ]
    results = compare_metas(metas, fields)

    # Prepare report
    out_path = Path(args.out) if args.out else (root / "step2_metadata_and_config_report.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines += ["# STEP2 report: OME-Zarr metadata comparison + PSF config check", ""]
    lines += [f"Root: {root}", f"OME-Zarr files found: {len(zarr_dirs)}", f"Reference for config compare: {ref}", ""]
    lines += [f"Config dir: {cfg_dir}", f"Configs used: {[str(p) for p in cfg_paths] if cfg_paths else 'NONE FOUND'}", ""]
    lines += ["## Fields checked", ", ".join(fields), ""]

    identical = [r.field for r in results if r.identical]
    different = [r.field for r in results if not r.identical]
    lines += ["## Identical fields", ", ".join(identical) if identical else "-", ""]
    lines += ["## Different fields", ", ".join(different) if different else "-", ""]

    # Detailed differences
    for r in results:
        if r.identical:
            continue
        lines += ["", f"### Field: {r.field}"]
        for val, files in sorted(r.values_to_files.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"* value = {val} | files = {len(files)}")
            for f in files[:10]:
                lines.append(f"    - {f}")
            if len(files) > 10:
                lines.append(f"    - ... (+{len(files)-10} more)")

    # Config check section
    if cfg_paths:
        img_meta = extract_ome_zarr_meta_for_compare(ref)
        lines += ["", "## PSFGenerator configs"]
        lines += [
            f"Reference image axes={img_meta.get('axes')} shape={img_meta.get('shape')} "
            f"voxel_um={img_meta.get('voxel_size_um')} NA={img_meta.get('objective_na')}",
            "",
        ]

        for cfg_path in cfg_paths:
            cfg = parse_psfgenerator_config(cfg_path)
            ok, missing = check_config_has_required(cfg)
            lines += [f"### Config: {cfg_path.name}", f"Required keys present: {ok}"]
            if missing:
                lines += [f"Missing keys: {missing}"]
            lines += ["Key summary:"]
            for k in ["Lambda", "NA", "NX", "NY", "NZ", "ResAxial", "ResLateral", "Type", "Scale"]:
                lines.append(f"  - {k}: {_fmt(cfg.get(k))}")
            lines += ["Comparison to image metadata:"]
            lines += ["  - " + s for s in compare_config_to_image(cfg, img_meta)]
            lines += [""]

    # Compact per-file summary
    lines += ["", "## Per-file (compact)"]
    for f, m in metas.items():
        vx = m.get("voxel_size_um", {}) or {}
        lines.append(
            f"- {f}\n"
            f"    axes={m.get('axes')} shape={m.get('shape')} dtype={m.get('dtype')} chunks={m.get('chunks')}\n"
            f"    voxel_um(x,y,z)=({vx.get('x')},{vx.get('y')},{vx.get('z')})\n"
            f"    channels={m.get('channel_names')}\n"
            f"    NA={m.get('objective_na')} mag={m.get('objective_magnification')} immersion={m.get('immersion')}\n"
            f"    RI(imm,sample)=({m.get('refractive_index_immersion')},{m.get('refractive_index_sample')})\n"
            f"    modality={m.get('modality')} sim_mode={m.get('sim_mode')}\n"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())