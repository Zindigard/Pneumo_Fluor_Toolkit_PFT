from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Tuple

import napari
import zarr

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
)

#  dask for lazy diff computation
try:
    import dask.array as da

    _HAVE_DASK = True
except Exception:
    da = None
    _HAVE_DASK = False



def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "setup.cfg").exists():
            return p
        if (p / "src").exists():
            return p
    return start.parent


def discover_3d_omezarrs(project_root: Path) -> List[Path]:
    base = project_root / "results" / "img" / "3d_data"
    if not base.exists():
        return []
    return sorted([p for p in base.rglob("image.ome.zarr") if p.is_dir()])


def resolve_to_image_omezarr(path: Union[str, Path]) -> Path:
    """
    Accepts:
      - .../image.ome.zarr
      - a parent folder that contains image.ome.zarr somewhere inside
    Returns the resolved image.ome.zarr path or raises ValueError.
    """
    p = Path(str(path)).resolve()
    if p.is_dir() and p.name == "image.ome.zarr":
        return p

    if p.is_dir():
        hits = [h for h in p.rglob("image.ome.zarr") if h.is_dir()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(
                "Folder contains multiple image.ome.zarr. Please pick one:\n"
                + "\n".join(str(h) for h in hits[:10])
            )

    raise ValueError(f"Not an image.ome.zarr folder and none found inside:\n{p}")

def open_omezarr_array(zarr_dir: Path) -> zarr.Array:
    zarr_dir = resolve_to_image_omezarr(zarr_dir)
    root = zarr.open_group(str(zarr_dir), mode="r")
    if "0" not in root:
        raise KeyError(f"Zarr array '0' not found inside: {zarr_dir}")
    arr = root["0"]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (C,Z,Y,X). Got shape={arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"Expected >=3 channels. Got C={arr.shape[0]} in shape={arr.shape}")
    return arr


def as_lazy(x):
    """
    Make x lazy if dask is installed.

    IMPORTANT:
    - da.from_zarr expects a STORE/PATH, not a numpy array
    - da.from_array works for numpy arrays and zarr.Array (chunked)
    """
    if not _HAVE_DASK:
        return x

    chunks = getattr(x, "chunks", None)
    if chunks is None:
        return da.from_array(x, chunks="auto")
    return da.from_array(x, chunks=chunks, asarray=False)


@dataclass
class CompareState:
    path_a: Optional[Path] = None
    path_b: Optional[Path] = None
    arr_a: Optional[zarr.Array] = None
    arr_b: Optional[zarr.Array] = None
    selected_channel: int = 0
    loaded: bool = False
    roi_yx: Optional[Tuple[slice, slice]] = None  # (yslice, xslice)


class DropFilter(QWidget):
    """
    Intercepts drag/drop so napari does NOT try to open non-image files
    (like metadata_full.xml). We resolve dropped folders to image.ome.zarr.
    """

    def __init__(self, parent, on_drop_callback):
        super().__init__(parent)
        self._on_drop = on_drop_callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        if event.type() == QEvent.Drop:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
                if paths:
                    self._on_drop(paths)
                event.acceptProposedAction()
                return True
        return False


class OmeZarr3DCompareWidget(QWidget):
    def __init__(self, viewer: "napari.Viewer", project_root: Path):
        super().__init__()
        self.viewer = viewer
        self.project_root = project_root
        self.state = CompareState()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("3D OME-Zarr Compare (A | B | abs(A−B))"))

        self.layout().addWidget(QLabel("Image A:"))
        row_a = QHBoxLayout()
        self.combo_a = QComboBox()
        row_a.addWidget(self.combo_a)
        self.path_a_edit = QLineEdit()
        self.path_a_edit.setPlaceholderText("…or paste path to image.ome.zarr (or parent folder) for A")
        row_a.addWidget(self.path_a_edit)
        self.btn_a_browse = QPushButton("Browse A…")
        self.btn_a_browse.clicked.connect(lambda: self.on_browse(slot="A"))
        row_a.addWidget(self.btn_a_browse)
        self.layout().addLayout(row_a)

        self.layout().addWidget(QLabel("Image B:"))
        row_b = QHBoxLayout()
        self.combo_b = QComboBox()
        row_b.addWidget(self.combo_b)
        self.path_b_edit = QLineEdit()
        self.path_b_edit.setPlaceholderText("…or paste path to image.ome.zarr (or parent folder) for B")
        row_b.addWidget(self.path_b_edit)
        self.btn_b_browse = QPushButton("Browse B…")
        self.btn_b_browse.clicked.connect(lambda: self.on_browse(slot="B"))
        row_b.addWidget(self.btn_b_browse)
        self.layout().addLayout(row_b)

        row_actions = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh list")
        self.btn_refresh.clicked.connect(self.refresh_lists)
        row_actions.addWidget(self.btn_refresh)

        self.btn_load = QPushButton("Load A + B")
        self.btn_load.clicked.connect(self.load_both)
        row_actions.addWidget(self.btn_load)
        self.layout().addLayout(row_actions)

        self.layout().addWidget(QLabel("Crop ROI (draw 1 rectangle, applies to all panes):"))
        row_crop = QHBoxLayout()
        self.btn_crop_apply = QPushButton("Apply crop")
        self.btn_crop_apply.clicked.connect(self.apply_crop_from_rectangle)
        row_crop.addWidget(self.btn_crop_apply)

        self.btn_crop_reset = QPushButton("Reset crop")
        self.btn_crop_reset.clicked.connect(self.reset_crop)
        row_crop.addWidget(self.btn_crop_reset)
        self.layout().addLayout(row_crop)

        # --- info ---
        self.info = QLabel(
            "Controls:\n"
            "  • Mouse wheel / Z slider = scroll Z\n"
            "  • Q = channel 0 (Blue), W = channel 1 (Green), E = channel 2 (Red)\n"
            "Difference pane:\n"
            "  • abs(A−B) is GRAYSCALE (black=0 diff, brighter=more diff)\n"
            "Drag & drop:\n"
            "  • Drop image.ome.zarr OR a parent folder that contains it.\n"
            "  • Drop 1 folder => sets A, drop 2 folders => sets A and B.\n"
        )
        self.layout().addWidget(self.info)

        self.viewer.grid.enabled = True
        self.viewer.grid.spacing = 20  

        qt_viewer = self.viewer.window._qt_viewer
        qt_viewer.canvas.native.setStyleSheet("background-color: white;")

        self.roi_layer = self._ensure_roi_layer()

        self._register_hotkeys()

        self.refresh_lists()

        qt_viewer.setAcceptDrops(True)
        self._drop_filter = DropFilter(qt_viewer, self._handle_drop)
        qt_viewer.installEventFilter(self._drop_filter)

    
    def show_error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)

    def _pretty_name(self, p: Path) -> str:
        parts = p.parts
        if len(parts) >= 3:
            return "/".join(parts[-3:])
        return str(p)

    def _register_hotkeys(self) -> None:
        @self.viewer.bind_key("q")
        def _ch0(_viewer):
            self.select_channel(0)

        @self.viewer.bind_key("w")
        def _ch1(_viewer):
            self.select_channel(1)

        @self.viewer.bind_key("e")
        def _ch2(_viewer):
            self.select_channel(2)

    def _ensure_roi_layer(self):
        existing = self.viewer.layers.get("CROP_ROI", None)
        if existing is not None:
            return existing
        return self.viewer.add_shapes(
            name="CROP_ROI",
            shape_type="rectangle",
            edge_color="white",
            face_color=[0, 0, 0, 0],  # transparent
            edge_width=2,
        )

    def _clear_image_layers_keep_roi(self) -> None:
        keep_name = "CROP_ROI"
        for layer in list(self.viewer.layers):
            if layer.name != keep_name:
                self.viewer.layers.remove(layer)
        self.roi_layer = self._ensure_roi_layer()

    def refresh_lists(self) -> None:
        paths = discover_3d_omezarrs(self.project_root)

        def fill_combo(combo: QComboBox):
            combo.clear()
            combo.addItem("— select —", "")
            for p in paths:
                combo.addItem(self._pretty_name(p), str(p))

        fill_combo(self.combo_a)
        fill_combo(self.combo_b)

    def on_browse(self, slot: str) -> None:
        p = QFileDialog.getExistingDirectory(
            self, f"Select image.ome.zarr (or parent) for {slot}", str(self.project_root)
        )
        if not p:
            return
        if slot.upper() == "A":
            self.path_a_edit.setText(p)
        else:
            self.path_b_edit.setText(p)

    def _get_path_for_slot(self, slot: str) -> Optional[Path]:
        slot = slot.upper()
        if slot == "A":
            txt = self.path_a_edit.text().strip().strip('"')
            if txt:
                return Path(txt)
            data = self.combo_a.currentData()
            return Path(data) if data else None

        txt = self.path_b_edit.text().strip().strip('"')
        if txt:
            return Path(txt)
        data = self.combo_b.currentData()
        return Path(data) if data else None

    def _handle_drop(self, paths: List[Path]) -> None:
        try:
            resolved = [resolve_to_image_omezarr(p) for p in paths if p.exists()]
        except Exception as e:
            self.show_error("Drop failed", f"{type(e).__name__}: {e}")
            return

        if not resolved:
            return

        self.path_a_edit.setText(str(resolved[0]))
        if len(resolved) >= 2:
            self.path_b_edit.setText(str(resolved[1]))

    def load_both(self) -> None:
        path_a = self._get_path_for_slot("A")
        path_b = self._get_path_for_slot("B")

        if path_a is None or path_b is None:
            self.show_error("Missing input", "Select both Image A and Image B.")
            return

        try:
            arr_a = open_omezarr_array(path_a)
            arr_b = open_omezarr_array(path_b)
        except Exception as e:
            self.show_error("Open failed", f"{type(e).__name__}: {e}")
            return

        if arr_a.shape[1:] != arr_b.shape[1:]:
            self.show_error(
                "Shape mismatch",
                f"A shape={arr_a.shape} vs B shape={arr_b.shape}\n"
                "Z/Y/X must match to compute abs(A−B)."
            )
            return

        self.state.path_a = resolve_to_image_omezarr(path_a)
        self.state.path_b = resolve_to_image_omezarr(path_b)
        self.state.arr_a = arr_a
        self.state.arr_b = arr_b
        self.state.loaded = True

        self._clear_image_layers_keep_roi()
        self._add_layers_for_channel(self.state.selected_channel)

        self.viewer.dims.ndisplay = 2

    def apply_crop_from_rectangle(self) -> None:
        if not self.state.loaded:
            self.show_error("Not loaded", "Load Image A and B first, then draw a rectangle.")
            return

        self.roi_layer = self._ensure_roi_layer()
        if len(self.roi_layer.data) == 0:
            self.show_error("No ROI", "Draw a rectangle in the CROP_ROI layer first.")
            return

        rect = self.roi_layer.data[-1]
        ys = rect[:, 0]
        xs = rect[:, 1]

        y0 = int(max(0, min(ys)))
        y1 = int(max(0, max(ys)))
        x0 = int(max(0, min(xs)))
        x1 = int(max(0, max(xs)))

        if (y1 - y0) < 2 or (x1 - x0) < 2:
            self.show_error("ROI too small", "ROI must be at least 2×2 pixels.")
            return

        self.state.roi_yx = (slice(y0, y1), slice(x0, x1))
        self._add_layers_for_channel(self.state.selected_channel)

    def reset_crop(self) -> None:
        self.state.roi_yx = None
        if self.state.loaded:
            self._add_layers_for_channel(self.state.selected_channel)

    def _add_layers_for_channel(self, ch: int) -> None:
        if self.state.arr_a is None or self.state.arr_b is None:
            return

        self._clear_image_layers_keep_roi()

        a = self.state.arr_a[ch, :, :, :]
        b = self.state.arr_b[ch, :, :, :]

        if self.state.roi_yx is not None:
            ysl, xsl = self.state.roi_yx
            a = a[:, ysl, xsl]
            b = b[:, ysl, xsl]

        aL = as_lazy(a)
        bL = as_lazy(b)

        diff = aL - bL
        if _HAVE_DASK:
            diff = da.fabs(diff)
        else:
            import numpy as np

            diff = np.abs(diff)

        cmaps = {0: "blue", 1: "green", 2: "red"}
        cmap = cmaps.get(ch, "gray")

        layer_a = self.viewer.add_image(aL, name=f"A • C{ch}", colormap=cmap, blending="additive")
        layer_a.grid_position = (0, 0)

        layer_b = self.viewer.add_image(bL, name=f"B • C{ch}", colormap=cmap, blending="additive")
        layer_b.grid_position = (0, 1)

        # Absolute  is GRAYSCALE
        layer_d = self.viewer.add_image(diff, name=f"abs(A−B) • C{ch}", colormap="gray")
        layer_d.grid_position = (0, 2)

        self.viewer.grid.enabled = True

    def select_channel(self, ch: int) -> None:
        ch = int(ch)
        if ch not in (0, 1, 2):
            return
        self.state.selected_channel = ch
        if not self.state.loaded:
            return
        self._add_layers_for_channel(ch)


def main() -> None:
    here = Path(__file__).resolve()
    project_root = find_project_root(here)

    viewer = napari.Viewer(title="PFT • 3D OME-Zarr Compare Viewer")
    widget = OmeZarr3DCompareWidget(viewer, project_root=project_root)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


if __name__ == "__main__":
    main()

