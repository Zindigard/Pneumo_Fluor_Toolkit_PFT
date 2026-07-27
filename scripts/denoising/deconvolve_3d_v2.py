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

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile
import zarr

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root, ensure_dir
from PFT.core_prog_parts.denoising.deconvolution_no_fuji import (
    deconvolve_omezarr_3ch_to_omezarr_skimage,
    DEFAULT_CHANNEL_WAVELENGTH_NM,
)

"Applies scikit-image Richardson-Lucy deconvolution to 3D image data"


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"
DEFAULT_OUT_ROOT = DEFAULT_PROJECT_ROOT / "results" / "deconv"
DEFAULT_PANEL_MODES = ("rgb", "red", "green", "blue")


def list_image_folders(root_3d: Path) -> list[Path]:
    return [p.parent for p in sorted(root_3d.rglob("image.ome.zarr"))]


def available_pyramid_levels(image_omezarr: Path) -> list[int]:
    root = zarr.open_group(str(image_omezarr), mode="r")
    ms = root.attrs.get("multiscales")
    if not ms or not isinstance(ms, list) or not ms[0].get("datasets"):
        return [0]
    ds = ms[0]["datasets"]
    return list(range(len(ds)))


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


def prompt_float(prompt: str, *, default: float | None = None) -> float:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Please enter a valid number.")


def prompt_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    opts_lower = [o.lower() for o in options]
    while True:
        s = input(prompt).strip().lower()
        if s == "" and default is not None:
            return default
        if s in opts_lower:
            return options[opts_lower.index(s)]
        print(f"Choose one of: {', '.join(options)}")


def prompt_yesno(prompt: str, *, default: bool = False) -> bool:
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


def _open_level_array(zarr_path: Path, level: int):
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


def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    return _scale_to_u8(img, vmin, vmax)


def _compose_rgb(cyx_u8: np.ndarray, mode: str) -> np.ndarray:
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


def _add_header(panel_u8: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(panel_u8)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    title_height = 36
    w, h = img.size

    header = Image.new("RGB", (w, title_height), (0, 0, 0))
    draw = ImageDraw.Draw(header)
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)

    out = Image.new("RGB", (w, h + title_height))
    out.paste(header, (0, 0))
    out.paste(img, (0, title_height))
    return np.asarray(out)


def _make_comparison_panel_raw_scaled(
    raw_cyx: np.ndarray,
    dec_cyx: np.ndarray,
    *,
    mode: str = "rgb",
) -> np.ndarray:
    if raw_cyx.shape != dec_cyx.shape:
        raise ValueError(f"Shape mismatch raw={raw_cyx.shape}, dec={dec_cyx.shape}")

    c = raw_cyx.shape[0]
    raw_u8 = []
    dec_u8 = []

    for ch in range(c):
        vmin = float(np.min(raw_cyx[ch]))
        vmax = float(np.max(raw_cyx[ch]))
        raw_u8.append(_scale_to_u8(raw_cyx[ch], vmin, vmax))
        dec_u8.append(_scale_to_u8(dec_cyx[ch], vmin, vmax))

    raw_u8 = np.stack(raw_u8, axis=0)
    dec_u8 = np.stack(dec_u8, axis=0)

    raw_rgb = _compose_rgb(raw_u8, mode)
    dec_rgb = _compose_rgb(dec_u8, mode)
    return np.concatenate([raw_rgb, dec_rgb], axis=1)


def _make_comparison_panel_normalized(
    raw_cyx: np.ndarray,
    dec_cyx: np.ndarray,
    *,
    mode: str = "rgb",
) -> np.ndarray:
    if raw_cyx.shape != dec_cyx.shape:
        raise ValueError(f"Shape mismatch raw={raw_cyx.shape}, dec={dec_cyx.shape}")

    c = raw_cyx.shape[0]
    raw_u8 = []
    dec_u8 = []

    for ch in range(c):
        raw_u8.append(_normalize_to_u8(raw_cyx[ch]))
        dec_u8.append(_normalize_to_u8(dec_cyx[ch]))

    raw_u8 = np.stack(raw_u8, axis=0)
    dec_u8 = np.stack(dec_u8, axis=0)

    raw_rgb = _compose_rgb(raw_u8, mode)
    dec_rgb = _compose_rgb(dec_u8, mode)
    return np.concatenate([raw_rgb, dec_rgb], axis=1)


def save_slice_comparison_panels(
    *,
    original_zarr: Path,
    original_level: int,
    deconv_zarr: Path,
    deconv_level: int,
    out_dir: Path,
    modes: tuple[str, ...] = DEFAULT_PANEL_MODES,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_arr, raw_axes = _open_level_array(original_zarr, original_level)
    dec_arr, dec_axes = _open_level_array(deconv_zarr, deconv_level)

    if raw_arr.ndim != 4 or dec_arr.ndim != 4:
        raise ValueError(
            f"Expected 4D arrays for slice export. Got raw={raw_arr.shape}, dec={dec_arr.shape}"
        )

    c_raw, z_raw, y_raw, x_raw = raw_arr.shape
    c_dec, z_dec, y_dec, x_dec = dec_arr.shape
    if c_raw != c_dec or z_raw != z_dec or y_raw != y_dec or x_raw != x_dec:
        raise ValueError(
            "Original and deconvolved arrays differ in shape: "
            f"raw={raw_arr.shape}, deconv={dec_arr.shape}. "
            "Use matching levels/shapes for comparison export."
        )

    valid_modes = {"rgb", "red", "green", "blue"}
    bad_modes = [m for m in modes if m not in valid_modes]
    if bad_modes:
        raise ValueError(f"Unsupported panel mode(s): {bad_modes}. Valid: {sorted(valid_modes)}")

    print(f"\n[COMPARE] Saving slice panels to: {out_dir}")
    print(f"  raw    : {original_zarr} (L{original_level})")
    print(f"  deconv : {deconv_zarr} (L{deconv_level})")
    print(f"  shape  : {raw_arr.shape}")
    print(f"  modes  : {', '.join(modes)}")
    print("  variants: raw_scaled, normalized")

    saved_count = 0
    variants = ("raw_scaled", "normalized")

    for variant in variants:
        for mode in modes:
            mode_dir = out_dir / variant / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            for z in range(z_raw):
                raw_cyx = _read_z_slice_cyx(raw_arr, raw_axes, z)
                dec_cyx = _read_z_slice_cyx(dec_arr, dec_axes, z)

                if variant == "raw_scaled":
                    panel = _make_comparison_panel_raw_scaled(raw_cyx, dec_cyx, mode=mode)
                else:
                    panel = _make_comparison_panel_normalized(raw_cyx, dec_cyx, mode=mode)

                title = f"z={z:03d} | LEFT: original | RIGHT: deconvolved | {variant} | mode={mode}"
                panel = _add_header(panel, title)

                tifffile.imwrite(
                    str(mode_dir / f"z{z:03d}__original_vs_deconv__{variant}__{mode}.tif"),
                    panel,
                )
                saved_count += 1

    print(f"[COMPARE] Saved {saved_count} slice panels across {len(variants)} variant(s) and {len(modes)} mode(s).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive scikit-image Richardson-Lucy deconvolution for 3D OME-Zarr.")
    ap.add_argument("--root_3d", default=str(DEFAULT_3D_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))

    ap.add_argument("--folder", default=None, help="Folder containing image.ome.zarr (skip selection)")
    ap.add_argument("--level", type=int, default=None, help="OME-Zarr pyramid level (default: 2 if available else 0)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--model", choices=["BW", "GL", "RW"], default=None)
    ap.add_argument("--background", type=float, default=None)
    ap.add_argument("--clip", action="store_true", help="Pass clip=True to scikit-image RL")
    ap.add_argument("--filter_epsilon", type=float, default=None)

    args = ap.parse_args()

    root_3d = Path(args.root_3d)
    out_root = ensure_dir(Path(args.out_root))

    if args.folder:
        in_zarr = Path(args.folder) / "image.ome.zarr"
        if not in_zarr.exists():
            raise SystemExit(f"Missing: {in_zarr}")
    else:
        in_zarr = interactive_select_dataset(root_3d)

    levels = available_pyramid_levels(in_zarr)
    default_level = 2 if 2 in levels else 0
    if args.level is None:
        print(f"\nAvailable pyramid levels: {levels}")
        level = prompt_int(
            f"Choose level (default {default_level}): ",
            min_v=min(levels),
            max_v=max(levels),
            default=default_level,
        )
        if level not in levels:
            raise SystemExit(f"Invalid level {level}. Must be one of: {levels}")
    else:
        level = int(args.level)
        if level not in levels:
            raise SystemExit(f"Invalid level {level}. Must be one of: {levels}")

    model = args.model or prompt_choice("Choose PSF model [BW/GL/RW] (default BW): ", ["BW", "GL", "RW"], default="BW")
    iters = args.iters if args.iters is not None else prompt_int("Iterations (default 15): ", min_v=1, max_v=500, default=15)
    background = args.background if args.background is not None else prompt_float("Background (default 0.0): ", default=0.0)

    clip = bool(args.clip)
    if not clip:
        clip = prompt_yesno("Use clip=True for RL?", default=False)

    filter_epsilon = args.filter_epsilon
    if filter_epsilon is None:
        use_eps = prompt_yesno("Set filter_epsilon?", default=False)
        if use_eps:
            filter_epsilon = prompt_float("filter_epsilon value (example 1e-6): ", default=1e-6)

    export_panels = prompt_yesno("Save original vs deconvolved panel for every Z slice?", default=True)

    project_root = find_repo_root(Path(__file__).resolve())

    print("\nRun configuration:")
    print(f"  project root     : {project_root}")
    print(f"  input zarr       : {in_zarr}")
    print(f"  out root         : {out_root}")
    print(f"  level            : {level}")
    print(f"  PSF model        : {model}")
    print(f"  iterations       : {iters}")
    print(f"  background       : {background}")
    print(f"  clip             : {clip}")
    print(f"  filter_epsilon   : {filter_epsilon}")
    print(f"  save panels      : {export_panels}")
    if export_panels:
        print(f"  panel modes      : {', '.join(DEFAULT_PANEL_MODES)}")

    info = deconvolve_omezarr_3ch_to_omezarr_skimage(
        in_omezarr=in_zarr,
        out_root=out_root,
        model=model,
        iters=iters,
        background=background,
        level=level,
        channel_wavelength_nm=dict(DEFAULT_CHANNEL_WAVELENGTH_NM),
        overwrite=True,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    if export_panels:
        compare_dir = ensure_dir(info.out_dir / "slice_comparison_panels")
        save_slice_comparison_panels(
            original_zarr=in_zarr,
            original_level=level,
            deconv_zarr=info.out_zarr,
            deconv_level=0,
            out_dir=compare_dir,
            modes=DEFAULT_PANEL_MODES,
        )

    print("\nDONE")
    print(f"  output zarr : {info.out_zarr}")
    print(f"  outdir      : {info.out_dir}")
    if export_panels:
        print(f"  panel dir   : {info.out_dir / 'slice_comparison_panels'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())