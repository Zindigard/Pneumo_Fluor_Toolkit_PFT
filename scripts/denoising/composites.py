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
import re
import sys

_THIS_FILE = Path(__file__).resolve()

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile
import zarr

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root, ensure_dir


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"
DEFAULT_DECONV_ROOT = DEFAULT_PROJECT_ROOT / "results" / "deconv"
DEFAULT_MODE = "rgb"


def list_image_folders(root_3d: Path) -> list[Path]:
    return [p.parent for p in sorted(root_3d.rglob("image.ome.zarr"))]


def prompt_int(prompt: str, *, min_v: int | None = None, max_v: int | None = None, default: int | None = None) -> int:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            v = default
        else:
            try:
                v = int(s)
            except ValueError:
                print("Please enter an integer.")
                continue

        if min_v is not None and v < min_v:
            print(f"Value must be >= {min_v}")
            continue
        if max_v is not None and v > max_v:
            print(f"Value must be <= {max_v}")
            continue
        return v


def prompt_yesno(prompt: str, *, default: bool = True) -> bool:
    default_s = "y" if default else "n"
    while True:
        s = input(f"{prompt} [y/n] (default {default_s}): ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please answer y or n.")


def interactive_select_dataset(root_3d: Path) -> Path:
    folders = list_image_folders(root_3d)
    if not folders:
        raise SystemExit(f"No image.ome.zarr found under: {root_3d}")

    print("\nAvailable 3D datasets:")
    for i, f in enumerate(folders):
        rel = f.relative_to(root_3d) if root_3d in f.parents else f
        print(f"  [{i}] {rel}")

    idx = prompt_int(
        f"\nSelect dataset index [0-{len(folders)-1}]: ",
        min_v=0,
        max_v=len(folders) - 1,
    )
    in_zarr = folders[idx] / "image.ome.zarr"
    if not in_zarr.exists():
        raise SystemExit(f"Missing: {in_zarr}")
    return in_zarr


def _get_axes(zarr_path: Path) -> str:
    root = zarr.open_group(str(zarr_path), mode="r")
    ms = root.attrs.get("multiscales")
    if ms and isinstance(ms, list) and ms[0].get("axes"):
        axes = ms[0]["axes"]
        if isinstance(axes, list):
            return "".join(a["name"] if isinstance(a, dict) else str(a) for a in axes).lower()
        return str(axes).lower()
    return "czyx"


def _open_level_array(zarr_path: Path, level: int = 0):
    root = zarr.open_group(str(zarr_path), mode="r")
    arr = root[str(level)]
    axes = _get_axes(zarr_path)
    return arr, axes


def _read_z_slice_cyx(arr, axes: str, z: int, t: int = 0) -> np.ndarray:
    axes = axes.lower()
    sel = []
    for ax in axes:
        if ax == "t":
            sel.append(t)
        elif ax == "c":
            sel.append(slice(None))
        elif ax == "z":
            sel.append(z)
        elif ax in ("y", "x"):
            sel.append(slice(None))
        else:
            sel.append(0)

    slab = np.asarray(arr[tuple(sel)], dtype=np.float32)
    rem_axes = "".join(a for a in axes if a not in ("t", "z"))

    if rem_axes != "cyx":
        try:
            perm = [rem_axes.index("c"), rem_axes.index("y"), rem_axes.index("x")]
            slab = np.transpose(slab, perm)
        except ValueError as exc:
            raise ValueError(f"Unsupported remaining axes order '{rem_axes}' after slicing.") from exc
    return slab


def _scale_to_u8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    x = (img - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _compose_rgb(cyx_u8: np.ndarray, mode: str = "rgb") -> np.ndarray:
    c, y, x = cyx_u8.shape
    out = np.zeros((y, x, 3), dtype=np.uint8)

    ch0 = cyx_u8[0] if c > 0 else np.zeros((y, x), dtype=np.uint8)
    ch1 = cyx_u8[1] if c > 1 else np.zeros((y, x), dtype=np.uint8)
    ch2 = cyx_u8[2] if c > 2 else np.zeros((y, x), dtype=np.uint8)

    if mode == "rgb":
        out[..., 0] = ch2
        out[..., 1] = ch1
        out[..., 2] = ch0
    elif mode == "blue":
        out[..., 2] = ch0
    elif mode == "green":
        out[..., 1] = ch1
    elif mode == "red":
        out[..., 0] = ch2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out


def _add_header(panel_u8: np.ndarray, text: str, title_height: int = 34) -> np.ndarray:
    img = Image.fromarray(panel_u8)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    w, h = img.size
    header = Image.new("RGB", (w, title_height), (0, 0, 0))
    draw = ImageDraw.Draw(header)
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)

    out = Image.new("RGB", (w, h + title_height))
    out.paste(header, (0, 0))
    out.paste(img, (0, title_height))
    return np.asarray(out)


def _tile_images_horizontally(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile.")

    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    if len(set(heights)) != 1:
        raise ValueError(f"All tiles must have same height. Got heights={heights}")

    canvas = np.zeros((heights[0], sum(widths), 3), dtype=np.uint8)
    x0 = 0
    for img in images:
        w = img.shape[1]
        canvas[:, x0:x0 + w, :] = img
        x0 += w
    return canvas


def _extract_iteration_from_path(path: Path) -> int | None:
    parts = [path.name] + [p.name for p in path.parents[:4]]
    text = " ".join(parts)

    patterns = [
        r"(?:^|[_\-\s])iters?[_\-\s]?(\d+)(?:$|[_\-\s])",
        r"(?:^|[_\-\s])iter[_\-\s]?(\d+)(?:$|[_\-\s])",
        r"(?:^|[_\-\s])it[_\-\s]?(\d+)(?:$|[_\-\s])",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def find_iteration_zarrs_for_dataset(
    *,
    deconv_root: Path,
    dataset_name: str,
) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []

    for zarr_path in sorted(deconv_root.rglob("image.ome.zarr")):
        text = str(zarr_path).lower()
        if dataset_name.lower() not in text:
            continue

        it = _extract_iteration_from_path(zarr_path)
        if it is None:
            continue

        candidates.append((it, zarr_path))

    unique: dict[int, Path] = {}
    for it, p in candidates:
        unique[it] = p

    return sorted(unique.items(), key=lambda t: t[0])


def build_iteration_panel_for_one_slice(
    *,
    raw_cyx: np.ndarray,
    iter_cyx_list: list[tuple[int, np.ndarray]],
    mode: str = "rgb",
) -> np.ndarray:
    c = raw_cyx.shape[0]

    raw_u8 = []
    for ch in range(c):
        vmin = float(np.min(raw_cyx[ch]))
        vmax = float(np.max(raw_cyx[ch]))
        raw_u8.append(_scale_to_u8(raw_cyx[ch], vmin, vmax))
    raw_u8 = np.stack(raw_u8, axis=0)

    tiles = []
    raw_rgb = _compose_rgb(raw_u8, mode=mode)
    tiles.append(_add_header(raw_rgb, "raw"))

    for it, arr_cyx in iter_cyx_list:
        if arr_cyx.shape != raw_cyx.shape:
            raise ValueError(f"Shape mismatch for iteration {it}: raw={raw_cyx.shape}, iter={arr_cyx.shape}")

        iter_u8 = []
        for ch in range(c):
            vmin = float(np.min(raw_cyx[ch]))
            vmax = float(np.max(raw_cyx[ch]))
            iter_u8.append(_scale_to_u8(arr_cyx[ch], vmin, vmax))
        iter_u8 = np.stack(iter_u8, axis=0)

        tile_rgb = _compose_rgb(iter_u8, mode=mode)
        tiles.append(_add_header(tile_rgb, f"iter {it}"))

    return _tile_images_horizontally(tiles)


def save_all_iteration_composites(
    *,
    original_zarr: Path,
    deconv_root: Path,
    out_dir: Path,
    mode: str = DEFAULT_MODE,
    level_original: int = 0,
    level_deconv: int = 0,
) -> None:
    dataset_name = original_zarr.parent.name
    iter_zarrs = find_iteration_zarrs_for_dataset(
        deconv_root=deconv_root,
        dataset_name=dataset_name,
    )
    if not iter_zarrs:
        raise SystemExit(
            f"No deconvolved iteration zarrs found for dataset '{dataset_name}' under: {deconv_root}\n"
            f"Expected folders/files containing the dataset name and iteration number."
        )

    out_dir = ensure_dir(out_dir)

    raw_arr, raw_axes = _open_level_array(original_zarr, level=level_original)
    if raw_arr.ndim != 4:
        raise ValueError(f"Expected 4D original array, got {raw_arr.shape}")

    loaded_iter_arrays = []
    for it, zarr_path in iter_zarrs:
        arr, axes = _open_level_array(zarr_path, level=level_deconv)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D deconvolved array for iter {it}, got {arr.shape}")
        loaded_iter_arrays.append((it, zarr_path, arr, axes))

    z_count = raw_arr.shape[1]
    print("\nFound iteration outputs:")
    for it, p, arr, _axes in loaded_iter_arrays:
        print(f"  iter {it:>3}: {p}")
        print(f"            shape={arr.shape}")

    print(f"\nSaving huge per-slice composites to: {out_dir}")
    print(f"Dataset : {dataset_name}")
    print(f"Slices  : {z_count}")
    print(f"Mode    : {mode}")
    print("Columns : raw + all available iterations")

    for z in range(z_count):
        raw_cyx = _read_z_slice_cyx(raw_arr, raw_axes, z)

        iter_cyx_list = []
        for it, _path, arr, axes in loaded_iter_arrays:
            iter_cyx = _read_z_slice_cyx(arr, axes, z)
            iter_cyx_list.append((it, iter_cyx))

        panel = build_iteration_panel_for_one_slice(
            raw_cyx=raw_cyx,
            iter_cyx_list=iter_cyx_list,
            mode=mode,
        )
        panel = _add_header(panel, f"{dataset_name} | z={z:03d} | raw + all iterations", title_height=38)

        tifffile.imwrite(
            str(out_dir / f"z{z:03d}__all_iterations_composite__{mode}.tif"),
            panel,
        )

    summary_lines = [
        f"dataset: {dataset_name}",
        f"original: {original_zarr}",
        f"mode: {mode}",
        f"columns: raw + " + ", ".join(f"iter {it}" for it, *_ in loaded_iter_arrays),
        f"files: {z_count}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nDONE")


def main() -> int:
    project_root = find_repo_root(Path(__file__).resolve())
    root_3d = DEFAULT_3D_ROOT if DEFAULT_3D_ROOT.exists() else project_root / "results" / "img" / "3d_data"
    deconv_root = DEFAULT_DECONV_ROOT if DEFAULT_DECONV_ROOT.exists() else project_root / "results" / "deconv"

    original_zarr = interactive_select_dataset(root_3d)
    dataset_name = original_zarr.parent.name

    print(f"\nSelected dataset: {dataset_name}")
    print(f"Original zarr   : {original_zarr}")
    print(f"Deconv root     : {deconv_root}")

    mode = "rgb"
    use_alt = prompt_yesno("Use RGB composite mode? (No = red/green/blue-only is not implemented here, keep RGB)", default=True)
    if not use_alt:
        print("Keeping mode='rgb'.")

    out_dir = ensure_dir(deconv_root / "analysis" / dataset_name / "all_iteration_composites")

    save_all_iteration_composites(
        original_zarr=original_zarr,
        deconv_root=deconv_root,
        out_dir=out_dir,
        mode=mode,
        level_original=0,
        level_deconv=0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
