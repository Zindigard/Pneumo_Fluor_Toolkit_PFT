from __future__ import annotations
from pathlib import Path

from PFT.core_prog_parts.save import export_2d

def test_export_2d_creates_outputs(tmp_path, arr_cyx, meta_2ch):
    out_dir = export_2d(
        arr=arr_cyx,
        meta=meta_2ch,              
        dataset_name="test_dataset",
        preview_mode="wga_dapi",
        out_base=tmp_path,
        visualize=False,
        save_preview_png=True,
        scalebar_um=4.0,
        wga_ch=0,
        dapi_ch=1,
        save_omezarr=False,         
    )

    out_dir = Path(out_dir)
    assert (out_dir / "metadata.txt").exists()
    assert (out_dir / "metadata.xml").exists()  
    assert (out_dir / "preview.png").exists()
