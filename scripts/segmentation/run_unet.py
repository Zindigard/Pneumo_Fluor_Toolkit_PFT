"Runs a trained U-Net model on input images and saves the predicted masks and overlays"


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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PFT.core_prog_parts import visualization as viz
import tensorflow as tf


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find project root by walking upward until one of these markers exists:
      - pyproject.toml
      - .git
      - src/PFT
    """
    start = (start or Path(__file__)).resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "src" / "PFT").exists():
            return p
  
    return _pft_project_root(Path(__file__).resolve())


REPO_ROOT = find_repo_root(Path(__file__).resolve())


import sys
src_dir = REPO_ROOT / "src"

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.u_train import (
    dice_coef,
    dice_loss,
    bce_dice_loss,
    iou_coef,
)


@dataclass
class InferConfig:
    repo_root: Path = REPO_ROOT

    dataset: str = "2d_time"
    level: int = 0

    patch: int = 256
    normalize: str = "scale_uint16"   
    threshold: float = 0.5

    process_all: bool = False
    preview_only: bool = False

    # output
    out_root: Path = None
    model_path: Path = None


def yes_no_prompt(text, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        s = input(f"{text} {suffix} ").strip().lower()
        if s == "":
            return default
        if s in {"y", "yes"}:
            return True
        if s in {"n", "no"}:
            return False
        print("Please answer y or n.")


def choose_dataset_terminal(default="2d_time"):
    datasets = ["2d_time", "2d_wga_dapi"]
    print("\nChoose dataset/model:")
    for i, ds in enumerate(datasets, 1):
        tag = " (default)" if ds == default else ""
        print(f"  {i}) {ds}{tag}")

    s = input(f"Choose number or press Enter for [{default}]: ").strip()
    if s == "":
        return default
    try:
        k = int(s)
        if 1 <= k <= len(datasets):
            return datasets[k - 1]
    except ValueError:
        pass

    print(f"Invalid choice, using default: {default}")
    return default


def choose_mode_terminal():
    print("\nChoose processing mode:")
    print("  1) Process ONE OME-Zarr")
    print("  2) Process ALL OME-Zarr files in dataset folder")
    s = input("Choose number [1]: ").strip()
    if s == "2":
        return "all"
    return "one"


def choose_from_paths(paths: list[Path], title="Choose file"):
    if not paths:
        raise RuntimeError(f"No paths available for: {title}")

    print(f"\n{title}:")
    for i, p in enumerate(paths, 1):
        print(f"  {i}) {p}")

    while True:
        s = input("Choose number: ").strip()
        try:
            k = int(s)
            if 1 <= k <= len(paths):
                return paths[k - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def get_model_path(repo_root: Path, dataset: str) -> Path:
    return repo_root / "models" / f"u_net_{dataset}" / f"u_net_{dataset}_best.keras"


def get_output_root(repo_root: Path, dataset: str) -> Path:
    return repo_root / "results" / "U-net" / dataset


def get_default_input_root(repo_root: Path, dataset: str) -> Path:
    """
    Adjust this if your OME-Zarr files are stored elsewhere.
    """
    if dataset == "2d_time":
        return repo_root / "results" / "img" / "2d_data" / "2d_time"
    if dataset == "2d_wga_dapi":
        return repo_root / "results" / "img" / "2d_data" / "2d_wga_dapi"
    return repo_root / "results" / "img"


def find_omezarr_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*.ome.zarr"):
        if p.is_dir():
            out.append(p)
    return sorted(out)


def normalize_axes(axes: str) -> str:
    return axes.strip().lower()


def move_axis_to_end(arr: np.ndarray, src: int) -> np.ndarray:
    return np.moveaxis(arr, src, -1)


def normalize_crop_numpy(x: np.ndarray, mode: str):
    x = x.astype(np.float32)

    if mode == "scale_uint16":
        x = x / 65535.0
        x = np.clip(x, 0.0, 1.0)
        return x

    if mode == "percentile":
        out = np.empty_like(x, dtype=np.float32)
        for c in range(x.shape[-1]):
            xc = x[..., c]
            p1 = np.percentile(xc, 1)
            p2 = np.percentile(xc, 99.8)
            out[..., c] = (xc - p1) / (p2 - p1 + 1e-8)
        out = np.clip(out, 0.0, 1.0)
        return out

    raise ValueError(f"Unknown normalize mode: {mode}")


def image_to_rgb_uint8(img: np.ndarray, normalize_mode="percentile"):
    """
    img expected as (H, W, C)
    """
    x = normalize_crop_numpy(img, normalize_mode)

    if x.shape[-1] == 1:
        g = (np.clip(x[..., 0], 0, 1) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    if x.shape[-1] == 2:
        # channel 0 -> blue, channel 1 -> green
        b = (np.clip(x[..., 0], 0, 1) * 255).astype(np.uint8)
        g = (np.clip(x[..., 1], 0, 1) * 255).astype(np.uint8)
        r = np.zeros_like(g, dtype=np.uint8)
        return np.dstack([r, g, b])

    rgb = (np.clip(x[..., :3], 0, 1) * 255).astype(np.uint8)
    return rgb


def ensure_hwc_2d(arr2d_or_3d: np.ndarray, dataset: str) -> np.ndarray:
    """
    Convert a 2D plane into (H, W, C) as expected by model.
    """
    if arr2d_or_3d.ndim == 2:
        return arr2d_or_3d[..., None]

    if arr2d_or_3d.ndim == 3:
        # if likely CYX, 
        if arr2d_or_3d.shape[0] in (1, 2, 3, 4) and arr2d_or_3d.shape[-1] not in (1, 2, 3, 4):
            return np.moveaxis(arr2d_or_3d, 0, -1)

        # already HWC
        return arr2d_or_3d

    raise ValueError(f"Expected 2D plane or 2D multi-channel plane, got shape={arr2d_or_3d.shape}")


def iter_2d_planes(arr: np.ndarray, axes: str):
    """
    Yield (index_dict, plane_hwc) for supported axis layouts:
      yx
      cyx
      tyx
      tcyx
    """
    axes = normalize_axes(axes)
    arr = np.asarray(arr)

    if axes == "yx":
        yield {}, ensure_hwc_2d(arr, dataset="2d_time")
        return

    if axes == "cyx":
        yield {}, ensure_hwc_2d(arr, dataset="2d_wga_dapi")
        return

    if axes == "tyx":
        for t in range(arr.shape[0]):
            yield {"t": t}, ensure_hwc_2d(arr[t], dataset="2d_time")
        return

    if axes == "tcyx":
        for t in range(arr.shape[0]):
            yield {"t": t}, ensure_hwc_2d(arr[t], dataset="2d_wga_dapi")
        return

    raise ValueError(
        f"Unsupported axes '{axes}'. "
        f"This inference script currently supports yx, cyx, tyx, tcyx."
    )


def rebuild_mask_array(masks: list[np.ndarray], original_shape: tuple[int, ...], axes: str) -> np.ndarray:
    """
    Rebuild predicted mask array to same logical axes as input,
    but mask is integer/binary.
    Supported:
      yx   -> (Y, X)
      cyx  -> (Y, X)  [single binary mask from multi-channel input]
      tyx  -> (T, Y, X)
      tcyx -> (T, Y, X)
    """
    axes = normalize_axes(axes)

    if axes in ("yx", "cyx"):
        return masks[0].astype(np.uint8)

    if axes in ("tyx", "tcyx"):
        return np.stack(masks, axis=0).astype(np.uint8)

    raise ValueError(f"Unsupported axes for rebuild: {axes}")


def output_mask_axes(input_axes: str) -> str:
    """
    Input:
      yx   -> yx
      cyx  -> yx
      tyx  -> tyx
      tcyx -> tyx
    """
    input_axes = normalize_axes(input_axes)
    if input_axes == "yx":
        return "yx"
    if input_axes == "cyx":
        return "yx"
    if input_axes == "tyx":
        return "tyx"
    if input_axes == "tcyx":
        return "tyx"
    raise ValueError(f"Unsupported input axes: {input_axes}")


def pad_to_patch(img: np.ndarray, patch: int):
    """
    img: (H, W, C)
    """
    h, w, c = img.shape
    pad_h = (patch - (h % patch)) % patch
    pad_w = (patch - (w % patch)) % patch

    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)

    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return img_pad, (pad_h, pad_w)


def predict_mask_tiled(model, img_hwc: np.ndarray, cfg: InferConfig) -> np.ndarray:
    """
    Run whole-image prediction by non-overlapping tiles.
    """
    img_norm = normalize_crop_numpy(img_hwc, cfg.normalize)
    img_pad, (pad_h, pad_w) = pad_to_patch(img_norm, cfg.patch)

    H, W, C = img_pad.shape
    pred = np.zeros((H, W), dtype=np.float32)

    tiles = []
    coords = []

    for y0 in range(0, H, cfg.patch):
        for x0 in range(0, W, cfg.patch):
            tile = img_pad[y0:y0 + cfg.patch, x0:x0 + cfg.patch, :]
            tiles.append(tile)
            coords.append((y0, x0))

    batch = np.stack(tiles, axis=0).astype(np.float32)
    pred_batch = model.predict(batch, verbose=0)

    for k, (y0, x0) in enumerate(coords):
        pred[y0:y0 + cfg.patch, x0:x0 + cfg.patch] = pred_batch[k, ..., 0]

    if pad_h > 0:
        pred = pred[:-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :-pad_w]

    mask = (pred >= cfg.threshold).astype(np.uint8)
    return mask

def mask_to_outline(mask: np.ndarray) -> np.ndarray:
    """
    Simple outline from binary mask without scipy/skimage dependency.
    """
    m = mask.astype(bool)
    up = np.roll(m, -1, axis=0)
    down = np.roll(m, 1, axis=0)
    left = np.roll(m, 1, axis=1)
    right = np.roll(m, -1, axis=1)

    eroded_like = m & up & down & left & right
    outline = m & (~eroded_like)

    outline[0, :] = m[0, :]
    outline[-1, :] = m[-1, :]
    outline[:, 0] = m[:, 0]
    outline[:, -1] = m[:, -1]
    return outline.astype(np.uint8)


def save_overlay_pngs(
    plane_infos: list[tuple[dict[str, int], np.ndarray, np.ndarray]],
    out_dir: Path,
    dataset: str,
    normalize_mode: str,
):
    """
    plane_infos: list of (index_dict, plane_hwc, mask_yx)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx_info, img_hwc, mask_yx in plane_infos:
        rgb = image_to_rgb_uint8(img_hwc, normalize_mode)
        outline = mask_to_outline(mask_yx)

        overlay = rgb.copy()
        overlay[outline.astype(bool), 0] = 255
        overlay[outline.astype(bool), 1] = 255
        overlay[outline.astype(bool), 2] = 0

        suffix = ""
        if "t" in idx_info:
            suffix += f"_t{idx_info['t']:03d}"

        viz.save_unet_overlay_panel(
            rgb,
            mask_yx,
            overlay,
            out_dir / f"u_net_overlay{suffix}.png",
        )


def try_load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    custom_objects = {
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss,
        "iou_coef": iou_coef,
    }
    return tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)


def process_one_omezarr(zarr_path: Path, cfg: InferConfig, model) -> Path:
    print(f"\n=== PROCESSING ===")
    print(f"Input OME-Zarr: {zarr_path}")
    print(f"Dataset/model:  {cfg.dataset}")

    arr, axes = load_ome_zarr(zarr_path, level=cfg.level, as_numpy=True)
    axes = normalize_axes(axes)

    print(f"Loaded shape:    {arr.shape}")
    print(f"Loaded axes:     {axes}")

    plane_infos = []
    masks = []

    for idx_info, plane_hwc in iter_2d_planes(arr, axes):
        print(f"  Predicting plane {idx_info if idx_info else '{single}'} | image shape: {plane_hwc.shape}")
        mask_yx = predict_mask_tiled(model, plane_hwc, cfg)
        masks.append(mask_yx)
        plane_infos.append((idx_info, plane_hwc, mask_yx))

    mask_arr = rebuild_mask_array(masks, arr.shape, axes)
    mask_axes = output_mask_axes(axes)

    rel_name = zarr_path.stem.replace(".ome", "")
    out_dir = cfg.out_root / rel_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mask_zarr = out_dir / "pred_mask.ome.zarr"
    save_ome_zarr(
        out_mask_zarr,
        mask_arr.astype(np.uint8),
        mask_axes,
        meta=None,
        overwrite=True,
        chunks=None,
        pyramid_3d=False,
        pyramid_max_layer=0,
        pyramid_downscale=2,
    )

    save_overlay_pngs(
        plane_infos=plane_infos,
        out_dir=out_dir,
        dataset=cfg.dataset,
        normalize_mode=cfg.normalize,
    )

    print(f"Saved mask OME-Zarr: {out_mask_zarr}")
    print(f"Saved PNG overlays:  {out_dir}")
    return out_dir


def parse_args(cfg: InferConfig):
    import argparse

    p = argparse.ArgumentParser(description="Apply trained U-Net to OME-Zarr images.")
    p.add_argument("--dataset", default=None, choices=["2d_time", "2d_wga_dapi"])
    p.add_argument("--level", type=int, default=cfg.level)
    p.add_argument("--patch", type=int, default=cfg.patch)
    p.add_argument("--normalize", default=cfg.normalize, choices=["scale_uint16", "percentile"])
    p.add_argument("--threshold", type=float, default=cfg.threshold)
    p.add_argument("--all", action="store_true", dest="process_all")
    p.add_argument("--preview_only", action="store_true")
    p.add_argument("--input_root", default=None)
    p.add_argument("--input_path", default=None)
    args = p.parse_args()

    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    if args.dataset is None:
        cfg.dataset = choose_dataset_terminal(default=cfg.dataset)

    cfg.model_path = get_model_path(cfg.repo_root, cfg.dataset)
    cfg.out_root = get_output_root(cfg.repo_root, cfg.dataset)
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    return cfg, args


def main():
    cfg, args = parse_args(InferConfig())

    print("\n=== SETTINGS ===")
    print(f"Repo root:      {cfg.repo_root}")
    print(f"Dataset:        {cfg.dataset}")
    print(f"Model:          {cfg.model_path}")
    print(f"Output root:    {cfg.out_root}")
    print(f"Patch size:     {cfg.patch}")
    print(f"Normalize:      {cfg.normalize}")
    print(f"Threshold:      {cfg.threshold}")
    print(f"OME-Zarr level: {cfg.level}")

    model = try_load_model(cfg.model_path)

    if args.input_path is not None:
        targets = [Path(args.input_path)]
    else:
        input_root = Path(args.input_root) if args.input_root else get_default_input_root(cfg.repo_root, cfg.dataset)
        all_zarr = find_omezarr_dirs(input_root)

        if not all_zarr:
            raise RuntimeError(
                f"No .ome.zarr directories found under: {input_root}\n"
                f"Pass --input_root or --input_path explicitly."
            )

        if cfg.process_all:
            targets = all_zarr
        else:
            mode = choose_mode_terminal()
            if mode == "all":
                targets = all_zarr
            else:
                targets = [choose_from_paths(all_zarr, title="Choose ONE OME-Zarr to process")]

    print(f"\nNumber of files to process: {len(targets)}")
    for i, p in enumerate(targets, 1):
        print(f"  [{i:02d}] {p}")

    for p in targets:
        process_one_omezarr(p, cfg, model)

    print("\nFinished.")


if __name__ == "__main__":
    main()