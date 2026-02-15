from __future__ import annotations

from pathlib import Path
import numpy as np

from PFT.core_prog_parts.io import CziMeta, read_czi_xml
from PFT.core_prog_parts import visualize_2d


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
        preview.png
        image.ome.zarr
    Notes
    
    """
    if out_base is None:
        out_base = results_img_dir()

    stem = Path(meta.source_path).stem
    out_dir = out_base / dataset_name / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    write_metadata_txt_xml(out_dir, meta)

    if save_omezarr:
        try:
            from PFT.core_prog_parts.Ome_Zarr import save_ome_zarr_next_to_outputs

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

    #Acquisition 
    lines += [f"Dataset / sample ID: {fmt(getattr(meta, 'sample_id', None))}"]
    lines += [f"Title: {fmt(getattr(meta, 'title', None))}"]
    lines += [f"Date/time: {fmt(getattr(meta, 'creation_datetime', None))}"]
    lines += [f"Microscope / system: {fmt(getattr(meta, 'microscope_system', None))}"]
    lines += [f"Operator / user: {fmt(getattr(meta, 'operator', None))}"]
    lines += [""]

    # Modality and reconstruction
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

    #Optics and detection
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
            from PFT.core_prog_parts.Ome_Zarr import save_ome_zarr_next_to_outputs

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
