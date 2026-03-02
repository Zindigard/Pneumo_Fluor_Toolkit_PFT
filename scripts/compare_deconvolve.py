from __future__ import annotations
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import zarr
import tifffile
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RAW_LEVEL = 2
DEFAULT_DECONV_LEVEL = 0
DEFAULT_Z_LIST = [22, 24, 26, 28, 30]

DEFAULT_RAW_ROOT = r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\3d_data"
DEFAULT_DECONV_ROOT = r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\deconv"
DEFAULT_ANALYSIS_ROOT = r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\deconv\analysis"

_DECONV_RE = re.compile(
    r"^(?P<base>.+?)__DL2_RL__PSF(?P<model>BW|GL|RW)__iter(?P<iter>\d+)__L(?P<lvl>\d+)$"
)


def ask_int(prompt, default=None):
    s = input(f"{prompt} (default {default}): ").strip()
    return default if not s else int(s)

def ask_str(prompt, default=None):
    s = input(f"{prompt} (default {default}): ").strip()
    return default if not s else s

def ask_yesno(prompt, default=False):
    d = "y" if default else "n"
    s = input(f"{prompt} [y/n] (default {d}): ").strip().lower()
    if not s:
        return default
    return s.startswith("y")

def get_axes(zarr_path: Path) -> str:
    grp = zarr.open(str(zarr_path), mode="r")
    if "multiscales" in grp.attrs:
        ms = grp.attrs["multiscales"]
        axes = ms[0]["axes"]
        if isinstance(axes, list):
            return "".join(a["name"] if isinstance(a, dict) else a for a in axes)
        return axes
    raise ValueError(f"Cannot determine axes for {zarr_path}")


def open_omezarr_array(zarr_path: str, level: int):
    grp = zarr.open(zarr_path, mode="r")
    arr = grp[str(level)]
    axes = get_axes(Path(zarr_path))
    return arr, axes


def read_slice(arr, axes: str, z: int, t: int = 0):
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

    slab = np.asarray(arr[tuple(sel)], dtype=np.float32)

    rem_axes = "".join([a for a in axes if a not in ("t", "z")])
    if rem_axes != "cyx":
        perm = [rem_axes.index("c"), rem_axes.index("y"), rem_axes.index("x")]
        slab = np.transpose(slab, perm)

    return slab  # (C,Y,X)


def scale_to_u8(img, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    x = (img - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


def compose_rgb(cyx, mode):
    C, Y, X = cyx.shape
    out = np.zeros((Y, X, 3), dtype=np.uint8)

    ch0 = cyx[0] if C > 0 else 0
    ch1 = cyx[1] if C > 1 else 0
    ch2 = cyx[2] if C > 2 else 0

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

    return out


def mosaic(a, b, c):
    return np.concatenate([a, b, c], axis=1)


def add_title(panel_u8, title_text):
    img = Image.fromarray(panel_u8)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    title_height = 35
    w, h = img.size

    title_img = Image.new("RGB", (w, title_height), (0, 0, 0))
    draw = ImageDraw.Draw(title_img)
    draw.text((10, 8), title_text, fill=(255, 255, 255), font=font)

    combined = Image.new("RGB", (w, h + title_height))
    combined.paste(title_img, (0, 0))
    combined.paste(img, (0, title_height))

    return np.array(combined)



@dataclass
class DeconvEntry:
    base: str
    model: str
    iter_n: int
    lvl: int
    zarr_path: Path


def discover_deconv(root: str) -> List[DeconvEntry]:
    entries = []
    for p in Path(root).iterdir():
        if not p.is_dir():
            continue
        m = _DECONV_RE.match(p.name)
        if not m:
            continue
        zarr_path = p / "image.ome.zarr"
        if not zarr_path.exists():
            continue
        entries.append(
            DeconvEntry(
                base=m.group("base"),
                model=m.group("model"),
                iter_n=int(m.group("iter")),
                lvl=int(m.group("lvl")),
                zarr_path=zarr_path,
            )
        )
    return sorted(entries, key=lambda e: (e.base, e.model, e.iter_n))

def main():

    print("=== Deconvolution comparison ===")

    deconv_root = ask_str("Deconvolution root", DEFAULT_DECONV_ROOT)
    raw_root = ask_str("RAW root", DEFAULT_RAW_ROOT)
    analysis_root = ask_str("Analysis root", DEFAULT_ANALYSIS_ROOT)

    entries = discover_deconv(deconv_root)
    if not entries:
        print("No deconvolved datasets found.")
        return

    bases = sorted(set(e.base for e in entries))
    for i, b in enumerate(bases):
        print(f"[{i}] {b}")

    idx = ask_int("Select dataset index", 0)
    base = bases[idx]

    base_entries = [e for e in entries if e.base == base]

    model = ask_str("Model [BW/GL/RW]", "BW")
    base_entries = [e for e in base_entries if e.model == model]

    iters = sorted(set(e.iter_n for e in base_entries))
    print("Available iterations:", iters)

    run_all = ask_yesno("Run ALL iterations?", False)
    iter_list = iters if run_all else [ask_int("Choose iteration", iters[0])]

    z_input = input(f"Z list (default {DEFAULT_Z_LIST}): ").strip()
    z_list = DEFAULT_Z_LIST if not z_input else [int(x) for x in z_input.split()]

    raw_path = list(Path(raw_root).glob(f"**/{base}/image.ome.zarr"))[0]

    for itn in iter_list:
        entry = [e for e in base_entries if e.iter_n == itn][0]

        print(f"\n=== iter={itn} model={model} ===")

        raw_arr, raw_axes = open_omezarr_array(str(raw_path), DEFAULT_RAW_LEVEL)
        dec_arr, dec_axes = open_omezarr_array(str(entry.zarr_path), DEFAULT_DECONV_LEVEL)

        out_dir = Path(analysis_root) / base / f"PSF{model}" / f"iter{itn}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for z in z_list:

            raw = read_slice(raw_arr, raw_axes, z)
            dec = read_slice(dec_arr, dec_axes, z)
        
            diff = np.abs(dec - raw)

            for scale_mode in ["rawscale", "perscale"]:

                if scale_mode == "rawscale":
                    vmin = raw.min(axis=(1,2))
                    vmax = raw.max(axis=(1,2))
                else:
                    vmin = raw.min(axis=(1,2))
                    vmax = raw.max(axis=(1,2))

                raw_u = np.stack([scale_to_u8(raw[c], vmin[c], vmax[c]) for c in range(raw.shape[0])])
                dec_u = np.stack([scale_to_u8(dec[c], vmin[c], vmax[c]) for c in range(dec.shape[0])])

                diff_gray = scale_to_u8(diff.mean(axis=0), 0, diff.max())

                for mode in ["rgb", "blue", "green", "red"]:

                    raw_rgb = compose_rgb(raw_u, mode)
                    dec_rgb = compose_rgb(dec_u, mode)
                    diff_rgb = np.stack([diff_gray]*3, axis=-1)

                    panel = mosaic(raw_rgb, dec_rgb, diff_rgb)

                    title = f"{base} | iter={itn} | z={z} | {mode} | {scale_mode} | RAW | DECONV | DIFF"
                    panel = add_title(panel, title)

                    name = f"z{z:03d}__{mode}__{scale_mode}__panel.tif"
                    tifffile.imwrite(str(out_dir / name), panel)

    print("\nDONE.")


if __name__ == "__main__":
    main()
