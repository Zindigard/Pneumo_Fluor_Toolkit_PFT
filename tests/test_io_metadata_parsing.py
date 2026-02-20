from __future__ import annotations
from pathlib import Path
from PFT.core_prog_parts.io import _parse_scaling_um, _parse_channel_names

def test_parse_scaling_um_from_sample_xml():
    xml = Path("metadata.xml").read_text(encoding="utf-8")  
    sx, sy, sz = _parse_scaling_um(xml)
    assert sx is None or sx > 0
    assert sy is None or sy > 0
    assert (sz is None) or (sz > 0)

def test_parse_channel_names_from_sample_xml():
    xml = Path("metadata.xml").read_text(encoding="utf-8")
    names = _parse_channel_names(xml)
    assert names is None or isinstance(names, list)
    if names:
        assert all(isinstance(n, str) and n for n in names)
