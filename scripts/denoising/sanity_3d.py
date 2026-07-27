from __future__ import annotations
# Permit direct execution from a source checkout after scripts were grouped into subpackages.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()
for _pft_parent in [_PFT_SCRIPT_FILE.parent, *_PFT_SCRIPT_FILE.parents]:
    _pft_src = _pft_parent / "src"
    if (_pft_src / "PFT").exists():
        if str(_pft_src) not in _pft_sys.path:
            _pft_sys.path.insert(0, str(_pft_src))
        break

def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Locate the repository root independently of script nesting depth."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "PFT").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return _PFTPath.cwd()


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
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader
except Exception as e:
        raise SystemExit(
            "Missing deps. Install:\n"
            "  pip install ome-zarr zarr numcodecs dask\n\n"
            f"Import error: {e}"
        )

"""Sanity check for 3D OME-Zarr reading and metadata parsing using ome-zarr library."""

OMEZARR_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\3d_data")


def _json_preview(obj: Any, max_chars: int = 2500) -> str:
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False)
    except TypeError:
        s = json.dumps(str(obj), indent=2, ensure_ascii=False)
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... (truncated) ...")


def _infer_axes_from_ndim(ndim: int | None) -> str:
    if ndim == 2:
        return "yx"
    if ndim == 3:
        return "cyx"
    if ndim == 4:
        return "tcyx"
    if ndim == 5:
        return "tczyx"
    return "unknown"


def _get_axes_and_multiscales(node_meta: Dict[str, Any], ndim: int | None) -> Tuple[str, Any]:
    axes = None
    ms = None
    try:
        ms = node_meta.get("multiscales", [])
        if ms and isinstance(ms, list) and isinstance(ms[0], dict) and "axes" in ms[0]:
            axes_entries = ms[0]["axes"]
            axes = "".join(a["name"] if isinstance(a, dict) else str(a) for a in axes_entries)
    except Exception:
        axes = None

    if axes is None:
        axes = _infer_axes_from_ndim(ndim)

    return axes.strip().lower(), ms


def _count_channels(shape: Tuple[int, ...], axes: str) -> int | None:
    axes = axes.lower()
    if "c" not in axes:
        return None
    ci = axes.index("c")
    if ci >= len(shape):
        return None
    return int(shape[ci])


def _print_zarr_tree(g, indent: str = "") -> None:
    for name, subgroup in g.groups():
        print(f"{indent}[group] {name}/")
        _print_zarr_tree(subgroup, indent + "  ")

    for name, arr in g.arrays():
        compressor = getattr(arr, "compressor", None)
        comp_str = str(compressor) if compressor is not None else "-"
        chunks = getattr(arr, "chunks", None)
        print(
            f"{indent}[array] {name} | shape={arr.shape} | chunks={chunks} | "
            f"dtype={arr.dtype} | compressor={comp_str}"
        )


def find_omezarr_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    found = [p for p in root.rglob("*.ome.zarr") if p.is_dir()]
    found.sort(key=lambda p: (p.name != "image.ome.zarr", str(p).lower()))
    return found


def choose_from_list(title: str, items: List[str], default: int = 0) -> int:
    print("\n" + title)
    for i, it in enumerate(items):
        d = " (default)" if i == default else ""
        print(f"  [{i:02d}] {it}{d}")
    s = input("Choose number: ").strip()
    if s == "":
        return default
    idx = int(s)
    if idx < 0 or idx >= len(items):
        raise ValueError("Invalid selection.")
    return idx


def prompt_int(prompt: str, default: int = 0) -> int:
    s = input(f"{prompt} (empty={default}): ").strip()
    return int(s) if s else default


def inspect_omezarr(zarr_dir: Path, level: int = 0) -> None:
    try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader
    except Exception as e:
        raise SystemExit(
            "Missing deps. Install:\n"
            "  pip install ome-zarr zarr numcodecs dask\n\n"
            f"Import error: {e}"
        )

    zarr_dir = zarr_dir.expanduser().resolve()
    if not zarr_dir.exists():
        raise SystemExit(f"Not found: {zarr_dir}")

    print("=" * 90)
    print(f"OME-Zarr path: {zarr_dir}")
    print("=" * 90)

    store = parse_url(str(zarr_dir), mode="r").store
    root = zarr.open_group(store=store, mode="r")

    print("\nZarr tree (groups/arrays)")
    def _print_tree(g, indent=""):
        for name, subgroup in g.groups():
            _print_tree(subgroup, indent + "")
        for name, arr in g.arrays():
            compressor = getattr(arr, "compressor", None)
            comp_str = str(compressor) if compressor is not None else "-"
            chunks = getattr(arr, "chunks", None)
            print(
                f"[array] {name} | shape={arr.shape} | chunks={chunks} | "
                f"dtype={arr.dtype} | compressor={comp_str}"
            )
    _print_tree(root)

    loc = parse_url(str(zarr_dir), mode="r")
    reader = Reader(loc)
    nodes = list(reader())
    if not nodes:
        print("\nOME-NGFF node summary")
        print("No image nodes found by ome_zarr.Reader().")
        return

    node = nodes[0]
    data = node.data[level]
    shape = getattr(data, "shape", None)
    ndim = getattr(data, "ndim", None)

    root_ms = root.attrs.get("multiscales", None)

    def _infer_axes_from_multiscales(ms, fallback_ndim):
        if isinstance(ms, list) and ms and isinstance(ms[0], dict) and "axes" in ms[0]:
            axes_entries = ms[0]["axes"]
            axes = "".join(a["name"] if isinstance(a, dict) else str(a) for a in axes_entries)
            return axes.strip().lower()
        
        if fallback_ndim == 2: return "yx"
        if fallback_ndim == 3: return "cyx"
        if fallback_ndim == 4: return "czyx"  # treat 4D as TCYX since many 4D NGFFs are actually 3D+channel
        if fallback_ndim == 5: return "tczyx"
        return "unknown"

    axes = _infer_axes_from_multiscales(root_ms, ndim)

    def _count_channels(shape_tuple, axes_str):
        axes_str = axes_str.lower()
        if shape_tuple is None or "c" not in axes_str:
            return None
        ci = axes_str.index("c")
        if ci >= len(shape_tuple):
            return None
        return int(shape_tuple[ci])

    n_ch = _count_channels(tuple(shape) if shape is not None else None, axes)

    print("\nOME-NGFF node summary")
    print(f"Pyramid level: {level}")
    print(f"Shape: {shape}")
    print(f"ndim: {ndim}")
    print(f"Axes: {axes}")
    print(f"n_channels (from axes/shape): {n_ch if n_ch is not None else '-'}")

    # channel names
    ch_names = root.attrs.get("channel_names", None)
    if isinstance(ch_names, list) and ch_names:
        print(f"channel_names (root.attrs['channel_names']): {ch_names}")
        if n_ch is not None and len(ch_names) != n_ch:
            print(f"WARNING: len(channel_names)={len(ch_names)} but inferred n_channels={n_ch}")

   
    pft = root.attrs.get("pft_meta", None)
    if isinstance(pft, dict):
        pft_compact = dict(pft)
        if "raw_xml" in pft_compact:
            pft_compact["raw_xml"] = "<omitted>"
        print("\npft_meta")
        print(_json_preview(pft_compact, max_chars=2500))

    
    if "source_path" in root.attrs:
        print("\nsource_path")
        print(root.attrs["source_path"])

    
    print("\nmultiscales")
    print(_json_preview(root_ms if root_ms is not None else [], max_chars=2500))

    print("\nDone.")



def main() -> None:
    print(f"\nScanning for *.ome.zarr in:\n  {OMEZARR_ROOT}\n")

    zarr_candidates = find_omezarr_dirs(OMEZARR_ROOT)
    if not zarr_candidates:
        print("No *.ome.zarr folders found.")
        return

    items = [str(p) for p in zarr_candidates]
    idx = choose_from_list("Select OME-Zarr to inspect:", items, default=0)
    zarr_dir = zarr_candidates[idx]

    level = prompt_int("Select pyramid level", default=0)
    inspect_omezarr(zarr_dir, level=level)


if __name__ == "__main__":
    main()
