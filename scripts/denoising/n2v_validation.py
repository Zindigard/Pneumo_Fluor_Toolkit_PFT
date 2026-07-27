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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import matplotlib  
from n2v.models import N2V
from PFT.core_prog_parts.common_paths import ensure_dir
from PFT.core_prog_parts.decoder_omezar import ome_zarr_to_n2v_2d_stack
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr


"checks and visualizes the quality of Noise2Void denoising results."
_PLT = None
_RectangleSelector = None

def get_plt(gui: bool):
    global _PLT, _RectangleSelector
    if _PLT is not None:
        return _PLT, _RectangleSelector

    if gui:
        matplotlib.use("TkAgg", force=True)
    else:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    _PLT = plt
    _RectangleSelector = RectangleSelector
    return _PLT, _RectangleSelector


def project_root_from_this_file() -> Path:
    """Return the project root using the shared path helper."""
    return find_repo_root(_THIS_FILE)



def is_omezarr_store(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / ".zgroup").exists() or (p / "zarr.json").exists() or (p / ".zattrs").exists()


def list_zarr_samples(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    direct = [p for p in folder.iterdir() if is_omezarr_store(p)]
    if direct:
        return sorted(direct)

    nested: list[Path] = []
    for sub in [p for p in folder.iterdir() if p.is_dir()]:
        nested.extend([p for p in sub.iterdir() if is_omezarr_store(p)])
    return sorted(nested)


def _read_json_if_exists(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_pixel_size_um_xy(zarr_dir: Path) -> Tuple[float, float]:
    attrs = _read_json_if_exists(zarr_dir / ".zattrs")
    if attrs is None:
        zj = _read_json_if_exists(zarr_dir / "zarr.json")
        if isinstance(zj, dict) and "attributes" in zj and isinstance(zj["attributes"], dict):
            attrs = zj["attributes"]

    if not isinstance(attrs, dict):
        return (1.0, 1.0)

    ms = attrs.get("multiscales", None)
    if not (isinstance(ms, list) and ms and isinstance(ms[0], dict)):
        return (1.0, 1.0)

    axes = ms[0].get("axes", None)
    datasets = ms[0].get("datasets", None)
    if not (isinstance(axes, list) and isinstance(datasets, list) and datasets):
        return (1.0, 1.0)

    axis_names: List[str] = []
    for a in axes:
        if isinstance(a, str):
            axis_names.append(a)
        elif isinstance(a, dict) and "name" in a:
            axis_names.append(str(a["name"]))
        else:
            axis_names.append("")

    ds0 = datasets[0] if isinstance(datasets[0], dict) else {}
    ct = ds0.get("coordinateTransformations", None)
    if not (isinstance(ct, list) and ct and isinstance(ct[0], dict)):
        return (1.0, 1.0)

    scale = ct[0].get("scale", None)
    if not (isinstance(scale, list) and len(scale) == len(axis_names)):
        return (1.0, 1.0)

    def _idx(name: str) -> Optional[int]:
        try:
            return axis_names.index(name)
        except ValueError:
            return None

    iy, ix = _idx("y"), _idx("x")
    if iy is None or ix is None:
        return (1.0, 1.0)

    try:
        py = float(scale[iy])
        px = float(scale[ix])
        if np.isfinite(py) and np.isfinite(px) and py > 0 and px > 0:
            return (py, px)
    except Exception:
        pass

    return (1.0, 1.0)


def choose_from_list(items: list[str], prompt: str) -> int:
    print(prompt)
    for i, s in enumerate(items, 1):
        print(f"  {i}) {s}")
    while True:
        x = input("Choose number: ").strip()
        try:
            k = int(x)
            if 1 <= k <= len(items):
                return k - 1
        except Exception:
            pass
        print("Invalid selection.")


def ask_int(prompt: str, minv: int, maxv: int, default: int) -> int:
    while True:
        s = input(f"{prompt} [{default}] ").strip()
        if s == "":
            return default
        try:
            v = int(s)
            if minv <= v <= maxv:
                return v
        except Exception:
            pass
        print(f"Enter integer in [{minv}, {maxv}] or press Enter.")


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    d = "Y/n" if default_yes else "y/N"
    while True:
        s = input(f"{prompt} ({d}): ").strip().lower()
        if s == "":
            return default_yes
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please enter y or n (or press Enter).")


def percentile_norm(img: np.ndarray, p_low=1.0, p_high=99.8) -> Tuple[np.ndarray, float, float]:
    x = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, (p_low, p_high))
    hi = max(hi, lo + 1e-8)
    out = (x - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out, float(lo), float(hi)


def abs_diff(raw: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.abs(den.astype(np.float32, copy=False) - raw.astype(np.float32, copy=False))


def blue_rgb(img01: np.ndarray) -> np.ndarray:
    rgb = np.zeros((img01.shape[0], img01.shape[1], 3), dtype=np.float32)
    rgb[..., 2] = img01
    return rgb


def green_rgb(img01: np.ndarray) -> np.ndarray:
    rgb = np.zeros((img01.shape[0], img01.shape[1], 3), dtype=np.float32)
    rgb[..., 1] = img01
    return rgb


def make_rgb_two_channel(yxc: np.ndarray, d_lohi: Tuple[float, float], w_lohi: Tuple[float, float]) -> np.ndarray:
    d = yxc[..., 0].astype(np.float32, copy=False)
    w = yxc[..., 1].astype(np.float32, copy=False)
    d_lo, d_hi = d_lohi
    w_lo, w_hi = w_lohi
    d01 = np.clip((d - d_lo) / (d_hi - d_lo + 1e-8), 0, 1)
    w01 = np.clip((w - w_lo) / (w_hi - w_lo + 1e-8), 0, 1)
    rgb = np.zeros((yxc.shape[0], yxc.shape[1], 3), dtype=np.float32)
    rgb[..., 2] = d01
    rgb[..., 1] = w01
    return rgb


def _clip_diff_vmax(diff2d: np.ndarray, q: float = 99.9) -> float:
    diff2d = np.asarray(diff2d, dtype=np.float32)
    vmax = float(np.percentile(diff2d, q)) if diff2d.size else 1.0
    return max(vmax, 1e-6)


BAR_NM = 4.0
BAR_UM = BAR_NM / 1000.0  # 0.004 µm

def add_scalebar(ax, px_um_x: float, img_shape_yx: Tuple[int, int], bar_um: float = BAR_UM) -> None:
    h, w = img_shape_yx
    bar_px = int(round(bar_um / max(px_um_x, 1e-12)))

    if bar_px < 2:
        bar_px = 2
        label_nm = bar_px * px_um_x * 1000.0
    else:
        label_nm = BAR_NM

    margin = max(10, w // 50)
    y = h - margin
    x0 = w - margin - bar_px
    x1 = w - margin

    ax.plot([x0, x1], [y, y], linewidth=4)
    ax.text((x0 + x1) / 2.0, y - margin * 0.6, f"{label_nm:g} nm",
            ha="center", va="bottom", fontsize=10)


def save_tiff_figure(fig, out_path: Path, dpi: int = 250) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, format="tiff")

def save_single_panel(plt, img, out_path: Path, *, title: str, add_bar: bool, px_um_x: float,
                      is_diff: bool = False, diff_vmax: Optional[float] = None) -> None:
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    if is_diff:
        vmax = float(diff_vmax) if diff_vmax is not None else float(np.max(img))
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
    else:
        ax.imshow(img)

    ax.set_title(title)
    ax.axis("off")
    if add_bar:
        add_scalebar(ax, px_um_x, (img.shape[0], img.shape[1]))
    save_tiff_figure(fig, out_path)
    plt.close(fig)


class Cropper:
    def __init__(self, plt, RectangleSelector, img, title: str):
        self.plt = plt
        self.roi = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        self.ax.imshow(img)
        self.ax.set_title(title)
        self.ax.axis("off")

        RectangleSelector(
            self.ax,
            onselect=self._on_select,
            useblit=True,
            button=[1],
            interactive=True,
            drag_from_anywhere=True,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        print("\n=== CROPPING CONTROLS ===")
        print("- Drag mouse to draw rectangle")
        print("- ENTER to confirm crop")
        print("- ESC to cancel (no crop)\n")

    def _on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        cmin = int(np.floor(min(x1, x2)))
        cmax = int(np.ceil(max(x1, x2)))
        rmin = int(np.floor(min(y1, y2)))
        rmax = int(np.ceil(max(y1, y2)))
        self.roi = (rmin, rmax, cmin, cmax)

    def _on_key(self, event):
        if event.key == "enter":
            self.plt.close(self.fig)
        elif event.key == "escape":
            self.roi = None
            self.plt.close(self.fig)

    def run(self):
        self.plt.show()
        return self.roi


def load_frames_nyx1(sample_zarr: Path, channel: int) -> np.ndarray:
    s = ome_zarr_to_n2v_2d_stack(sample_zarr, channel=channel, time=None, z=None, normalize=None)
    s = np.asarray(s)
    if s.ndim != 4 or s.shape[-1] != 1:
        raise ValueError(f"Expected (N,Y,X,1), got {s.shape}")
    return s.astype(np.float32, copy=False)


def load_joint_frames_yxc(sample_zarr: Path, c0: int = 0, c1: int = 1) -> np.ndarray:
    s0 = load_frames_nyx1(sample_zarr, channel=c0)
    s1 = load_frames_nyx1(sample_zarr, channel=c1)
    n = min(s0.shape[0], s1.shape[0])
    return np.concatenate([s0[:n, ...], s1[:n, ...]], axis=-1)


@dataclass
class Selection:
    dataset: str
    sample: Path
    mode: str
    model_name: str
    frame_index: int
    do_crop: bool  


def _clamp_roi(roi, shape_yx: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    rmin, rmax, cmin, cmax = roi
    H, W = shape_yx
    rmin = max(0, min(H - 1, rmin))
    rmax = max(rmin + 1, min(H, rmax))
    cmin = max(0, min(W - 1, cmin))
    cmax = max(cmin + 1, min(W, cmax))
    return (rmin, rmax, cmin, cmax)


def run_one(sel: Selection, root: Path, *, gui: bool) -> None:
    plt, RectangleSelector = get_plt(gui)

    models_base = root / "models"
    out_base = root / "results" / "N2V"

    analysis_base = out_base / "analysis" / sel.dataset / sel.sample.name / sel.mode
    denoised_base = out_base / sel.dataset / sel.sample.name / sel.mode / "denoised"
    ensure_dir(analysis_base)
    ensure_dir(denoised_base)

    _, px_um_x = get_pixel_size_um_xy(sel.sample)

    if sel.dataset == "2d_time":
        stack = load_frames_nyx1(sel.sample, channel=0)
    else:
        if sel.mode == "joint":
            stack = load_joint_frames_yxc(sel.sample, 0, 1)
        elif sel.mode == "blue_only":
            stack = load_frames_nyx1(sel.sample, channel=0)
        elif sel.mode == "green_only":
            stack = load_frames_nyx1(sel.sample, channel=1)
        else:
            raise RuntimeError(f"Unknown mode: {sel.mode}")

    fi = int(np.clip(sel.frame_index, 0, stack.shape[0] - 1))
    raw_for_model = stack[fi]

    # predict
    model = N2V(None, sel.model_name, basedir=str(models_base))
    den = model.predict(raw_for_model, axes="YXC")

    out_zarr = denoised_base / "image.ome.zarr"
    if den.shape[-1] == 1:
        save_ome_zarr(out_zarr, den[..., 0].astype(np.float32, copy=False), axes="yx", meta=None, overwrite=True)
    else:
        save_ome_zarr(out_zarr, np.transpose(den, (2, 0, 1)).astype(np.float32, copy=False),
                      axes="cyx", meta=None, overwrite=True)
    print(f"[SAVED] {sel.dataset} | {sel.sample.name} | {sel.mode} | denoised: {out_zarr}")

    if den.shape[-1] == 1:
        raw2 = raw_for_model[..., 0]
        den2 = den[..., 0]

        raw01, lo, hi = percentile_norm(raw2, 1, 99.8)
        den01 = np.clip((den2 - lo) / (hi - lo + 1e-8), 0, 1)

        if sel.mode in ("time_ch0", "blue_only"):
            raw_vis = blue_rgb(raw01)
            den_vis = blue_rgb(den01)
        else:
            raw_vis = green_rgb(raw01)
            den_vis = green_rgb(den01)

        diff = abs_diff(raw2, den2)
        diff_vmax = _clip_diff_vmax(diff, 99.9)

        roi = None
        if sel.do_crop:
            roi = Cropper(
                plt, RectangleSelector,
                raw_vis,
                title=f"{sel.sample.name} | {sel.dataset}/{sel.mode} | frame {fi} — crop (Enter/Esc)"
            ).run()
            roi = _clamp_roi(roi, raw2.shape)

        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.imshow(raw_vis); ax1.set_title("RAW (display-normalized)"); ax1.axis("off")
        add_scalebar(ax1, px_um_x, raw2.shape)

        ax2.imshow(den_vis); ax2.set_title("DENOISED (display-normalized)"); ax2.axis("off")
        add_scalebar(ax2, px_um_x, raw2.shape)

        ax3.imshow(diff, cmap="gray", vmin=0, vmax=diff_vmax)
        ax3.set_title(f"|D-R| abs (vmax=p99.9={diff_vmax:.3g})")
        ax3.axis("off")

        save_tiff_figure(fig, analysis_base / "panel_before_after_diff.tiff")
        plt.close(fig)

        save_single_panel(plt, raw_vis, analysis_base / "panel_RAW.tiff",
                          title="RAW (display-normalized)", add_bar=True, px_um_x=px_um_x)
        save_single_panel(plt, den_vis, analysis_base / "panel_DENOISED.tiff",
                          title="DENOISED (display-normalized)", add_bar=True, px_um_x=px_um_x)
        save_single_panel(plt, diff, analysis_base / "panel_DIFF_abs_gray.tiff",
                          title=f"|D-R| abs (vmax=p99.9={diff_vmax:.3g})",
                          add_bar=False, px_um_x=px_um_x, is_diff=True, diff_vmax=diff_vmax)

        if roi is not None:
            rmin, rmax, cmin, cmax = roi
            rawc = raw2[rmin:rmax, cmin:cmax]
            denc = den2[rmin:rmax, cmin:cmax]

            rawc01, lo_c, hi_c = percentile_norm(rawc, 1, 99.8)
            denc01 = np.clip((denc - lo_c) / (hi_c - lo_c + 1e-8), 0, 1)

            rawc_vis = blue_rgb(rawc01) if sel.mode in ("time_ch0", "blue_only") else green_rgb(rawc01)
            denc_vis = blue_rgb(denc01) if sel.mode in ("time_ch0", "blue_only") else green_rgb(denc01)

            diffc = abs_diff(rawc, denc)
            diffc_vmax = _clip_diff_vmax(diffc, 99.9)

            fig = plt.figure(figsize=(15, 5), constrained_layout=True)
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            ax1.imshow(rawc_vis); ax1.set_title("CROP RAW (display-normalized)"); ax1.axis("off")
            add_scalebar(ax1, px_um_x, rawc.shape)

            ax2.imshow(denc_vis); ax2.set_title("CROP DENOISED (display-normalized)"); ax2.axis("off")
            add_scalebar(ax2, px_um_x, rawc.shape)

            ax3.imshow(diffc, cmap="gray", vmin=0, vmax=diffc_vmax)
            ax3.set_title(f"CROP |D-R| abs (vmax=p99.9={diffc_vmax:.3g})")
            ax3.axis("off")

            save_tiff_figure(fig, analysis_base / "panel_crop_before_after_diff.tiff")
            plt.close(fig)

        print(f"[SAVED] analysis TIFFs: {analysis_base}")
        return

    raw_yxc = raw_for_model
    den_yxc = den

    dapi_raw = raw_yxc[..., 0]
    wga_raw  = raw_yxc[..., 1]
    dapi_den = den_yxc[..., 0]
    wga_den  = den_yxc[..., 1]

    _, d_lo, d_hi = percentile_norm(dapi_raw, 1, 99.8)
    _, w_lo, w_hi = percentile_norm(wga_raw, 1, 99.8)

    rgb_raw = make_rgb_two_channel(raw_yxc, (d_lo, d_hi), (w_lo, w_hi))
    rgb_den = make_rgb_two_channel(den_yxc, (d_lo, d_hi), (w_lo, w_hi))

    diff_over = 0.5 * (abs_diff(dapi_raw, dapi_den) + abs_diff(wga_raw, wga_den))
    diff_over_vmax = _clip_diff_vmax(diff_over, 99.9)

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(rgb_raw); ax1.set_title("RAW RGB (display-normalized)"); ax1.axis("off")
    add_scalebar(ax1, px_um_x, (raw_yxc.shape[0], raw_yxc.shape[1]))

    ax2.imshow(rgb_den); ax2.set_title("DENOISED RGB (display-normalized)"); ax2.axis("off")
    add_scalebar(ax2, px_um_x, (raw_yxc.shape[0], raw_yxc.shape[1]))

    ax3.imshow(diff_over, cmap="gray", vmin=0, vmax=diff_over_vmax)
    ax3.set_title(f"mean |D-R| abs (vmax=p99.9={diff_over_vmax:.3g})")
    ax3.axis("off")

    save_tiff_figure(fig, analysis_base / "panel_RGB_before_after_diff.tiff")
    plt.close(fig)

    save_single_panel(plt, rgb_raw, analysis_base / "panel_RGB_RAW.tiff",
                      title="RAW RGB (display-normalized)", add_bar=True, px_um_x=px_um_x)
    save_single_panel(plt, rgb_den, analysis_base / "panel_RGB_DENOISED.tiff",
                      title="DENOISED RGB (display-normalized)", add_bar=True, px_um_x=px_um_x)
    save_single_panel(plt, diff_over, analysis_base / "panel_RGB_DIFF_abs_gray.tiff",
                      title=f"mean |D-R| abs (vmax=p99.9={diff_over_vmax:.3g})",
                      add_bar=False, px_um_x=px_um_x, is_diff=True, diff_vmax=diff_over_vmax)

    print(f"[SAVED] analysis TIFFs: {analysis_base}")


def run_all_for_dataset(root: Path, dataset: str, mode: str, model_name: str, frame_index: int) -> None:
    samples = list_zarr_samples(root / "results" / "training_files" / dataset)
    if not samples:
        print(f"[WARN] No samples for {dataset}")
        return

    for s in samples:
        sel = Selection(dataset=dataset, sample=s, mode=mode, model_name=model_name,
                        frame_index=frame_index, do_crop=False)
        run_one(sel, root, gui=False)  # Agg backend


def run_one_interactive(root: Path) -> None:
    datasets = ["2d_time", "2d_wga_dapi"]
    dataset = datasets[choose_from_list(datasets, "Choose dataset:")]

    samples = list_zarr_samples(root / "results" / "training_files" / dataset)
    if not samples:
        raise SystemExit(f"[ERROR] No samples found for {dataset}")

    sample = samples[choose_from_list([p.name for p in samples], "Choose sample:")]

    if dataset == "2d_time":
        stack0 = load_frames_nyx1(sample, channel=0)
        frame_index = ask_int("Frame index", 0, stack0.shape[0] - 1, 0)
        do_crop = ask_yes_no("Do crop for this sample?", default_yes=True)
        sel = Selection(dataset=dataset, sample=sample, mode="time_ch0", model_name="n2v_2d_time_raw",
                        frame_index=frame_index, do_crop=do_crop)
        run_one(sel, root, gui=True)  # TkAgg backend
        return

    modes = [
        ("joint (2-ch RGB)", "joint", "n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw"),
        ("blue-only (DAPI ch0)", "blue_only", "n2v_2d_wga_dapi_DAPI_blue_raw"),
        ("green-only (WGA ch1)", "green_only", "n2v_2d_wga_dapi_WGA_green_raw"),
    ]
    mi = choose_from_list([m[0] for m in modes], "Choose mode/model:")
    _, mode, model_name = modes[mi]

    if mode == "joint":
        stack0 = load_joint_frames_yxc(sample, 0, 1)
    elif mode == "blue_only":
        stack0 = load_frames_nyx1(sample, channel=0)
    else:
        stack0 = load_frames_nyx1(sample, channel=1)

    frame_index = ask_int("Frame index", 0, stack0.shape[0] - 1, 0)
    do_crop = ask_yes_no("Do crop for this sample?", default_yes=True)

    sel = Selection(dataset=dataset, sample=sample, mode=mode, model_name=model_name,
                    frame_index=frame_index, do_crop=do_crop)
    run_one(sel, root, gui=True)


def main() -> int:
    root = project_root_from_this_file()

    main_actions = [
        "Run ONE sample (optional crop)",
        "Run ALL samples for ONE dataset (no crop)",
        "Run ALL samples for BOTH datasets (no crop)",
    ]
    action = choose_from_list(main_actions, "Select action:")

    if action == 0:
        run_one_interactive(root)
        return 0

    if action == 1:
        datasets = ["2d_time", "2d_wga_dapi"]
        dataset = datasets[choose_from_list(datasets, "Choose dataset to run ALL samples:")]

        if dataset == "2d_time":
            samples = list_zarr_samples(root / "results" / "training_files" / "2d_time")
            if not samples:
                print("[ERROR] No 2d_time samples.")
                return 1
            n0 = load_frames_nyx1(samples[0], channel=0).shape[0]
            frame = ask_int("Frame index (applies to ALL samples)", 0, n0 - 1, 0)
            run_all_for_dataset(root, "2d_time", "time_ch0", "n2v_2d_time_raw", frame)
            return 0

        modes = [
            ("joint (2-ch RGB)", "joint", "n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw"),
            ("blue-only (DAPI ch0)", "blue_only", "n2v_2d_wga_dapi_DAPI_blue_raw"),
            ("green-only (WGA ch1)", "green_only", "n2v_2d_wga_dapi_WGA_green_raw"),
        ]
        mi = choose_from_list([m[0] for m in modes], "Choose mode/model (applies to ALL samples):")
        _, mode, model_name = modes[mi]

        samples = list_zarr_samples(root / "results" / "training_files" / "2d_wga_dapi")
        if not samples:
            print("[ERROR] No 2d_wga_dapi samples.")
            return 1

        if mode == "joint":
            n0 = load_joint_frames_yxc(samples[0], 0, 1).shape[0]
        elif mode == "blue_only":
            n0 = load_frames_nyx1(samples[0], channel=0).shape[0]
        else:
            n0 = load_frames_nyx1(samples[0], channel=1).shape[0]

        frame = ask_int("Frame index (applies to ALL samples)", 0, n0 - 1, 0)
        run_all_for_dataset(root, "2d_wga_dapi", mode, model_name, frame)
        return 0

    print("\nRunning ALL samples for BOTH datasets (no crop).")

    time_samples = list_zarr_samples(root / "results" / "training_files" / "2d_time")
    if time_samples:
        n0 = load_frames_nyx1(time_samples[0], channel=0).shape[0]
        frame_time = ask_int("2d_time frame index (for ALL samples)", 0, n0 - 1, 0)
        run_all_for_dataset(root, "2d_time", "time_ch0", "n2v_2d_time_raw", frame_time)
    else:
        print("[WARN] No 2d_time samples.")

    wga_samples = list_zarr_samples(root / "results" / "training_files" / "2d_wga_dapi")
    if not wga_samples:
        print("[WARN] No 2d_wga_dapi samples.")
        return 0

    modes = [
        ("joint (2-ch RGB)", "joint", "n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw"),
        ("blue-only (DAPI ch0)", "blue_only", "n2v_2d_wga_dapi_DAPI_blue_raw"),
        ("green-only (WGA ch1)", "green_only", "n2v_2d_wga_dapi_WGA_green_raw"),
    ]
    mi = choose_from_list([m[0] for m in modes], "2d_wga_dapi: Choose mode/model (ALL samples):")
    _, mode, model_name = modes[mi]

    if mode == "joint":
        n0 = load_joint_frames_yxc(wga_samples[0], 0, 1).shape[0]
    elif mode == "blue_only":
        n0 = load_frames_nyx1(wga_samples[0], channel=0).shape[0]
    else:
        n0 = load_frames_nyx1(wga_samples[0], channel=1).shape[0]

    frame_wga = ask_int("2d_wga_dapi frame index (for ALL samples)", 0, n0 - 1, 0)
    run_all_for_dataset(root, "2d_wga_dapi", mode, model_name, frame_wga)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())