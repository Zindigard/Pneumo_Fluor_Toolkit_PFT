from __future__ import annotations
from pathlib import Path
import numpy as np
from PFT.core_prog_parts.io import CziMeta, read_czi_xml
from PFT.core_prog_parts import visualize_2d


def _results_img_dir() -> Path:
    # repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "results" / "img"


def _safe_axes_label(meta: CziMeta) -> str:
    # Store header axes string if available.
    return meta.axes if isinstance(meta.axes, str) and meta.axes else "unknown"


def write_metadata_txt_xml(out_dir: Path, meta: CziMeta) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # metadata.txt
    txt = "\n".join([
        "=== CZI META (dataclass str) ===",
        str(meta),
        "",
        f"axes_label_saved: {_safe_axes_label(meta)}",
    ])
    (out_dir / "metadata.txt").write_text(txt, encoding="utf-8")

    # metadata.xml (raw)
    xml = read_czi_xml(meta.source_path)
    if isinstance(xml, str) and xml.strip():
        (out_dir / "metadata.xml").write_text(xml, encoding="utf-8")
    else:
        (out_dir / "metadata.xml").write_text("No XML metadata available.\n", encoding="utf-8")

""""
def save_ome_zarr(out_zarr_dir: Path, arr: np.ndarray, meta: CziMeta) -> None:
   
    out_zarr_dir.mkdir(parents=True, exist_ok=True)

    # Local import keeps package usable without ome-zarr installed
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    axes_label = _safe_axes_label(meta)

    store = parse_url(str(out_zarr_dir), mode="w").store
    root = store.root()

    write_image(image=arr, group=root, axes=axes_label)

    # Put scale into attrs (viewers can use it)
    root.attrs["pixel_size_um"] = {
        "x": meta.pixel_size_um_x,
        "y": meta.pixel_size_um_y,
        "z": meta.pixel_size_um_z,
    }
    root.attrs["source_path"] = meta.source_path
    root.attrs["channel_names"] = meta.channel_names
"""

def export_2d(
    arr: np.ndarray,
    meta: CziMeta,
    dataset_name: str,
    preview_mode: str,   # "time_blue" or "wga_dapi"
    out_base: Path | None = None,
    visualize: bool = False,
    save_preview_png: bool = True,
    scalebar_um: float = 5.0,
    wga_ch: int = 0,
    dapi_ch: int = 1,
) -> Path:
    """
    Export 2D:
        metadata.txt
        metadata.xml
        image.ome.zarr/
        preview.png
    """
    if out_base is None:
        out_base = _results_img_dir()

    stem = Path(meta.source_path).stem
    out_dir = out_base / dataset_name / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metadata_txt_xml(out_dir, meta)
    #save_ome_zarr(out_dir / "image.ome.zarr", arr, meta)

    # Create RGB 
    title = f"{dataset_name} | {Path(meta.source_path).name}"
    if preview_mode == "wga_dapi":
        rgb = visualize_2d.rgb_wga_dapi(arr, wga_ch=wga_ch, dapi_ch=dapi_ch)
    else:
        rgb = visualize_2d.rgb_time_hada_blue(arr)

    if save_preview_png:
        visualize_2d.save_rgb_preview_png(
            rgb01=rgb,
            out_png=out_dir / "preview.png",
            title=title,
            meta=meta,
            scalebar_um=scalebar_um,
        )

    if visualize:
        visualize_2d.preview_rgb(rgb, title=title, meta=meta, scalebar_um=scalebar_um)
        import matplotlib.pyplot as plt
        plt.show()

    return out_dir


def export_3d_metadata_only(meta: CziMeta, dataset_name: str, out_base: Path | None = None) -> Path:
    """
    Export metadata only for 3D:
        metadata.txt
        metadata.xml
    """
    if out_base is None:
        out_base = _results_img_dir()

    stem = Path(meta.source_path).stem
    out_dir = out_base / dataset_name / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metadata_txt_xml(out_dir, meta)
    return out_dir
