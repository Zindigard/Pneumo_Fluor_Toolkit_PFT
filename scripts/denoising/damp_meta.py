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
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import czifile

"""Script to read CZI files from the datasets, extract metadata and save it in a structured way for inspection."""

@dataclass(frozen=True)
class Paths:
    data_2d_time: Path = Path(r"E:\2D_data_time")
    data_2d_wga_dapi: Path = Path(r"E:\2D_data_WGA_DAPI_DNA")
    data_3d: Path = Path(r"E:\20220225_HADA_NADA_TADA_40min")


PATHS = Paths()


def repo_results_img_dir() -> Path:
    repo_root = _pft_project_root(Path(__file__).resolve())
    return repo_root / "results" / "img"


def list_czi_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.czi"))
    if not files:
        raise FileNotFoundError(f"No .czi files found in: {folder}")
    return files


def pick_one_file(folder: Path) -> Path:
    return list_czi_files(folder)[0]


def read_czi_header_and_xml(czi_path: Path) -> tuple[str | None, tuple[int, ...] | None, str | None]:
    with czifile.CziFile(str(czi_path)) as czi:
        axes = getattr(czi, "axes", None)
        shape = getattr(czi, "shape", None)
        # raw metadata XML
        xml = None
        try:
            xml = czi.metadata()
        except Exception:
            xml = None
    return axes, shape, xml


def read_czi_array_squeezed(czi_path: Path) -> np.ndarray:
    with czifile.CziFile(str(czi_path)) as czi:
        arr = czi.asarray()
    return np.squeeze(arr)


def format_summary(
    dataset_name: str,
    czi_path: Path,
    axes_full: str | None,
    shape_full: tuple[int, ...] | None,
    arr: np.ndarray,
) -> str:
    lines: list[str] = []
    lines.append(f"=== Pneumo-Fluor-Toolkit | CZI METADATA DUMP ===")
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"File: {czi_path.name}")
    lines.append(f"Path: {czi_path}")
    lines.append("")
    lines.append("--- CZI header (as reported by czifile) ---")
    lines.append(f"axes:  {axes_full!r}")
    lines.append(f"shape: {shape_full!r}")
    lines.append("")
    lines.append("--- Loaded array (np.squeeze(czi.asarray())) ---")
    lines.append(f"array.shape: {arr.shape}")
    lines.append(f"array.ndim:  {arr.ndim}")
    lines.append(f"dtype:       {arr.dtype}")
    if arr.size:
        lines.append(f"min:         {float(arr.min())}")
        lines.append(f"max:         {float(arr.max())}")
    else:
        lines.append("array is empty (size=0)")
    lines.append("")
    return "\n".join(lines)


def dump_one(dataset_name: str, folder: Path, out_base: Path) -> None:
    czi_path = pick_one_file(folder)

    axes_full, shape_full, xml = read_czi_header_and_xml(czi_path)
    arr = read_czi_array_squeezed(czi_path)

    # Output folder:
    out_dir = out_base / "meta_dump" / dataset_name / czi_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = format_summary(dataset_name, czi_path, axes_full, shape_full, arr)

    # Write metadata.txt
    (out_dir / "metadata.txt").write_text(summary, encoding="utf-8")

    # Write raw XML
    if isinstance(xml, str) and xml.strip():
        (out_dir / "metadata.xml").write_text(xml, encoding="utf-8")
    else:
        (out_dir / "metadata.xml").write_text(
            "No XML metadata returned by czifile.CziFile.metadata().\n",
            encoding="utf-8",
        )

    print(summary)
    print(f"Saved to: {out_dir}")
    print("-" * 70)


def main() -> None:
    out_base = repo_results_img_dir()

    dump_one("2d_time", PATHS.data_2d_time, out_base)
    dump_one("2d_wga_dapi", PATHS.data_2d_wga_dapi, out_base)
    dump_one("3d", PATHS.data_3d, out_base)


if __name__ == "__main__":
    main()
