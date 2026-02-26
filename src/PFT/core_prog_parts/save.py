from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile

from PFT.core_prog_parts.io import CziMeta, read_czi_xml
from PFT.core_prog_parts import visualize_2d

"""
Helper functions to save metadata, previews and OME-Zarr files for 2D and 3D datasets,
with structured output directories and error handling for missing metadata or OME-Zarr saving issues.
"""


def results_img_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "results" / "img"


def safe_axes_label(meta: CziMeta) -> str:
    # Store header axes string.
    return meta.axes if isinstance(meta.axes, str) and meta.axes else "unknown"


def write_metadata_txt_xml(out_dir: Path, meta: CziMeta) -> None:
    """
    Write:
      - metadata.txt (dataclass str dump)
      - metadata.xml (raw xml)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    txt = "\n".join(
        [
            "=== CZI META (dataclass str) ===",
            str(meta),
            "",
            f"axes_label_saved: {safe_axes_label(meta)}",
        ]
    )
    (out_dir / "metadata.txt").write_text(txt, encoding="utf-8")

    xml = read_czi_xml(meta.source_path)
    if isinstance(xml, str) and xml.strip():
        (out_dir / "metadata.xml").write_text(xml, encoding="utf-8")
    else:
        (out_dir / "metadata.xml").write_text("No XML metadata available.\n", encoding="utf-8")


def _dtype_max(arr: np.ndarray) -> float:
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    m = float(np.nanmax(arr)) if arr.size else 1.0
    return m if m > 0 else 1.0


def _linear01(img: np.ndarray, denom: float) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    out = img / float(denom)
    return np.clip(out, 0.0, 1.0)


def _minmax01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if not a.size:
        return np.zeros_like(a, dtype=np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _rgb01_time_raw(arr: np.ndarray) -> np.ndarray:
    img2d = visualize_2d.max_project_to_2d(arr)
    b = _linear01(img2d, _dtype_max(arr))
    z = np.zeros_like(b)
    return np.dstack([z, z, b])


def _rgb01_time_norm(arr: np.ndarray) -> np.ndarray:
    return visualize_2d.rgb_time_hada_blue(arr)


def _rgb01_wga_dapi_raw(arr: np.ndarray) -> np.ndarray:
    """
    WGA/DAPI requested mapping:
      Channel 0 -> BLUE
      Channel 1 -> GREEN
    Raw look: linear scaling by dtype max.
    """
    ch_ax = visualize_2d.find_channel_axis(arr)
    blue = visualize_2d.get_channel_2d(arr, 0, ch_ax)   # ch0
    green = visualize_2d.get_channel_2d(arr, 1, ch_ax)  # ch1

    b = _linear01(blue, _dtype_max(arr))
    g = _linear01(green, _dtype_max(arr))
    r = np.zeros_like(g)
    return np.dstack([r, g, b])


def _rgb01_wga_dapi_norm(arr: np.ndarray) -> np.ndarray:
    """
    Same mapping:
      Channel 0 -> BLUE
      Channel 1 -> GREEN
    Normalized look: percentile normalization per channel.
    """
    ch_ax = visualize_2d.find_channel_axis(arr)
    blue = visualize_2d.get_channel_2d(arr, 0, ch_ax)   # ch0
    green = visualize_2d.get_channel_2d(arr, 1, ch_ax)  # ch1

    b = visualize_2d.normalize01(blue)
    g = visualize_2d.normalize01(green)
    r = np.zeros_like(g)
    return np.dstack([r, g, b])


def save_raw_and_normalized_tiffs_and_previews(
    arr: np.ndarray,
    out_dir: Path,
    *,
    preview_mode: str,          # "time_blue" or "wga_dapi"
    title: str,
    meta: CziMeta,
    scalebar_um: float,
    save_preview_png: bool,
) -> None:
    """
    Saves:
      - image_raw.tif          (original dtype)
      - image_norm16.tif       (min-max normalized to uint16)

    Plus:
      - image_raw_rgb.tif      (RGB uint16, requested channel colors)
      - image_norm16_rgb.tif   (RGB uint16, requested channel colors)

    Plus (if save_preview_png):
      - preview_raw.png        (raw-look RGB)
      - preview_norm.png       (normalized-look RGB)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(out_dir / "image_raw.tif", arr)

    a01 = _minmax01(arr)  # min-max per-array
    a16 = np.round(a01 * 65535.0).astype(np.uint16) if a01.size else np.zeros_like(arr, dtype=np.uint16)
    tifffile.imwrite(out_dir / "image_norm16.tif", a16)

    if preview_mode == "wga_dapi":
        rgb_raw01 = _rgb01_wga_dapi_raw(arr)
        rgb_norm01 = _rgb01_wga_dapi_norm(arr)
    else:
        rgb_raw01 = _rgb01_time_raw(arr)
        rgb_norm01 = _rgb01_time_norm(arr)

    rgb_raw16 = np.round(rgb_raw01 * 65535.0).astype(np.uint16)
    rgb_norm16 = np.round(rgb_norm01 * 65535.0).astype(np.uint16)

    tifffile.imwrite(out_dir / "image_raw_rgb.tif", rgb_raw16)
    tifffile.imwrite(out_dir / "image_norm16_rgb.tif", rgb_norm16)

    if save_preview_png:
        visualize_2d.save_rgb_preview_png(
            rgb01=rgb_raw01,
            out_png=out_dir / "preview_raw.png",
            title=title + " | preview RAW (linear)",
            meta=meta,
            scalebar_um=scalebar_um,
        )
        visualize_2d.save_rgb_preview_png(
            rgb01=rgb_norm01,
            out_png=out_dir / "preview_norm.png",
            title=title + " | preview NORM (percentile)",
            meta=meta,
            scalebar_um=scalebar_um,
        )


def export_2d(
    arr: np.ndarray,
    meta: CziMeta,
    dataset_name: str,
    preview_mode: str,  # "time_blue" or "wga_dapi"
    out_base: Path | None = None,
    visualize: bool = False,
    save_preview_png: bool = True,
    scalebar_um: float = 5.0,
    wga_ch: int = 0,   
    dapi_ch: int = 1,  
    *,
    save_omezarr: bool = True,
    overwrite_omezarr: bool = True,
) -> Path:
    """
    Export 2D:
        metadata.txt
        metadata.xml

        image_raw.tif
        image_norm16.tif

        image_raw_rgb.tif
        image_norm16_rgb.tif

        preview_raw.png
        preview_norm.png

        image.ome.zarr
    """
    if out_base is None:
        out_base = results_img_dir()

    stem = Path(meta.source_path).stem
    out_dir = out_base / dataset_name / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metadata_txt_xml(out_dir, meta)

    title = f"{dataset_name} | {Path(meta.source_path).name}"

    save_raw_and_normalized_tiffs_and_previews(
        arr=arr,
        out_dir=out_dir,
        preview_mode=preview_mode,
        title=title,
        meta=meta,
        scalebar_um=scalebar_um,
        save_preview_png=save_preview_png,
    )

    if save_omezarr:
        try:
            from PFT.core_prog_parts.ome_zarr import save_ome_zarr_next_to_outputs

            save_ome_zarr_next_to_outputs(
                out_dir=out_dir,
                arr=arr,
                meta=meta,
                overwrite=overwrite_omezarr,
            )
        except Exception as e:
            (out_dir / "omezarr_error.txt").write_text(
                f"{type(e).__name__}: {e}\n",
                encoding="utf-8",
            )

    if visualize:
        if preview_mode == "wga_dapi":
            rgb = _rgb01_wga_dapi_norm(arr)
        else:
            rgb = _rgb01_time_norm(arr)

        visualize_2d.preview_rgb(rgb, title=title, meta=meta, scalebar_um=scalebar_um)
        import matplotlib.pyplot as plt
        plt.show()

    return out_dir


def export_3d_metadata_only(meta: CziMeta, dataset_name: str, out_base: Path | None = None) -> Path:
    if out_base is None:
        out_base = results_img_dir()

    stem = Path(meta.source_path).stem
    out_dir = out_base / dataset_name / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metadata_txt_xml(out_dir, meta)
    return out_dir


def fmt(v: object) -> str:
    if v is None:
        return "-"
    s = str(v).strip()
    return s if s else "-"


def write_metadata_full_xml(out_dir: Path, meta: CziMeta) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_xml = getattr(meta, "raw_xml", None)
    if not (isinstance(raw_xml, str) and raw_xml.strip()):
        raw_xml = read_czi_xml(meta.source_path)

    if isinstance(raw_xml, str) and raw_xml.strip():
        (out_dir / "metadata_full.xml").write_text(raw_xml, encoding="utf-8")
    else:
        (out_dir / "metadata_full.xml").write_text("No XML metadata available.\n", encoding="utf-8")


def write_3d_metadata_report_txt(out_dir: Path, meta: CziMeta) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    vox = (
        getattr(meta, "pixel_size_um_x", None),
        getattr(meta, "pixel_size_um_y", None),
        getattr(meta, "pixel_size_um_z", None),
    )
    axes = getattr(meta, "axes", None)
    header_shape = getattr(meta, "header_shape", None)

    ch_info = getattr(meta, "channel_info", None) or []
    sim_settings = getattr(meta, "sim_settings", None)

    lines: list[str] = []
    lines += ["# PFT 3D metadata report", ""]

    lines += [f"Dataset / sample ID: {fmt(getattr(meta, 'sample_id', None))}"]
    lines += [f"Title: {fmt(getattr(meta, 'title', None))}"]
    lines += [f"Date/time: {fmt(getattr(meta, 'creation_datetime', None))}"]
    lines += [f"Microscope / system: {fmt(getattr(meta, 'microscope_system', None))}"]
    lines += [f"Operator / user: {fmt(getattr(meta, 'operator', None))}"]
    lines += [""]

    lines += [f"Modality: {fmt(getattr(meta, 'modality', None))}"]
    lines += [f"SIM type: {fmt(getattr(meta, 'sim_mode', None))}"]
    lines += [f"Reconstruction software: {fmt(getattr(meta, 'application_name', None))}"]
    lines += [f"Reconstruction version: {fmt(getattr(meta, 'application_version', None))}"]
    lines += [f"Exported image type (raw vs reconstructed): {fmt(getattr(meta, 'exported_image_type', None))}"]
    lines += [""]

    if isinstance(sim_settings, dict) and sim_settings:
        lines += ["Reconstruction settings (subset):"]
        wanted = [
            "OTFThreshold",
            "NotchFilterAmplitude",
            "NotchFilterVariance",
            "ApodizationWeight",
            "ApodizationSheppardWeight",
            "SectioningAmplitude",
            "Sharpness",
        ]
        for k in wanted:
            if k in sim_settings:
                lines += [f"  - {k}: {sim_settings[k]}"]
        other = [k for k in sim_settings.keys() if k not in wanted and k != "ZEN_Comment"]
        lines += [f"  - other_sim_keys_count: {len(other)}"]
        if "ZEN_Comment" in sim_settings:
            lines += ["", "ZEN comment:", sim_settings["ZEN_Comment"], ""]
    else:
        lines += ["Reconstruction settings: -", ""]

    # Sampling and image geometry
    lines += [f"CZI axes (header): {fmt(axes)}"]
    lines += [f"CZI shape (header): {fmt(header_shape)}"]
    lines += [f"Voxel size (µm): dx={fmt(vox[0])}, dy={fmt(vox[1])}, dz={fmt(vox[2])}"]
    lines += [f"Pixel type / bit depth: {fmt(getattr(meta, 'pixel_type', None))}"]
    lines += [""]

    # Optics and detection
    lines += [f"Objective model: {fmt(getattr(meta, 'objective_model', None))}"]
    lines += [f"Objective magnification: {fmt(getattr(meta, 'objective_magnification', None))}"]
    lines += [f"Objective NA: {fmt(getattr(meta, 'objective_na', None))}"]
    lines += [f"Immersion medium: {fmt(getattr(meta, 'immersion', None))}"]
    lines += [f"Refractive index (immersion): {fmt(getattr(meta, 'refractive_index_immersion', None))}"]
    lines += [f"Refractive index (sample/mount): {fmt(getattr(meta, 'refractive_index_sample', None))}"]
    lines += [f"Detector type: {fmt(getattr(meta, 'detector_type', None))}"]
    lines += [f"Detector model: {fmt(getattr(meta, 'detector_model', None))}"]
    lines += [""]

    # Channels
    if not ch_info:
        lines += ["- (no per-channel info found in XML)"]
    else:
        for i, ch in enumerate(ch_info):
            lines += [
                f"[{i}] name={fmt(ch.get('name'))} | exc={fmt(ch.get('excitation_nm'))} nm | "
                f"em={fmt(ch.get('emission_nm'))} nm | exposure={fmt(ch.get('exposure_s'))} s | "
                f"type={fmt(ch.get('channel_type'))}"
            ]

    (out_dir / "metadata_report.txt").write_text("\n".join(lines), encoding="utf-8")


def export_3d(
    arr: np.ndarray,
    meta: CziMeta,
    *,
    dataset_folder: str,
    out_base: Path | None = None,
    save_omezarr: bool = True,
    overwrite_omezarr: bool = True,
) -> Path:
    """
    Export 3D:
        - image.ome.zarr
        - metadata_report.txt
        - metadata_full.xml
    """
    if out_base is None:
        out_base = results_img_dir()

    stem = Path(meta.source_path).stem
    out_dir = out_base / "3d_data" / dataset_folder / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    write_3d_metadata_report_txt(out_dir, meta)
    write_metadata_full_xml(out_dir, meta)

    if save_omezarr:
        try:
            from PFT.core_prog_parts.ome_zarr import save_ome_zarr_next_to_outputs

            save_ome_zarr_next_to_outputs(
                out_dir=out_dir,
                arr=arr,
                meta=meta,
                overwrite=overwrite_omezarr,
            )
        except Exception as e:
            (out_dir / "omezarr_error.txt").write_text(
                f"{type(e).__name__}: {e}\n",
                encoding="utf-8",
            )

    return out_dir