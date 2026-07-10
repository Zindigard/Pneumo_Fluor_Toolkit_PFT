from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
)

from PFT.core_prog_parts.common_paths import find_project_root


class PFTResultsWidget(QWidget):
    """
    Placeholder napari widget for PFT statistical and quantitative outputs.

    This widget is intentionally minimal at this stage. It reserves the GUI slot
    for future measurements, statistics, plots, CSV export, and model comparison.
    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.project_root = find_project_root(Path(__file__).resolve())

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("PFT results / statistics"))

        self.info = QLabel(
            "This widget is reserved for future statistical output.\n\n"
            "Planned outputs:\n"
            "• cell count and mask area\n"
            "• mean / median intensity per channel\n"
            "• Dice / IoU when reference masks are available\n"
            "• intensity profile and AUC\n"
            "• t-test / ANOVA outputs where applicable\n"
            "• CSV / JSON export\n\n"
            "Current status: blank placeholder."
        )
        self.info.setWordWrap(True)
        self.layout().addWidget(self.info)

        row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh selected layer info")
        self.refresh_btn.clicked.connect(self.refresh_layer_info)
        row.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("Export placeholder report…")
        self.export_btn.clicked.connect(self.export_placeholder_report)
        row.addWidget(self.export_btn)
        self.layout().addLayout(row)

        self.layer_info = QLabel("No layer inspected yet.")
        self.layer_info.setWordWrap(True)
        self.layout().addWidget(self.layer_info)

    def _active_layer(self):
        if not self.viewer.layers:
            return None
        return self.viewer.layers.selection.active or self.viewer.layers[-1]

    def refresh_layer_info(self) -> None:
        layer = self._active_layer()
        if layer is None or not hasattr(layer, "data"):
            self.layer_info.setText("No active image or label layer selected.")
            return

        data = layer.data
        shape = getattr(data, "shape", None)
        dtype = getattr(data, "dtype", None)
        metadata = getattr(layer, "metadata", {}) or {}
        step = metadata.get("pft_step", "not specified")
        axes = metadata.get("pft_layer_axes") or metadata.get("pft_axes") or "not specified"

        self.layer_info.setText(
            f"Selected layer: {layer.name}\n"
            f"Shape: {shape}\n"
            f"Dtype: {dtype}\n"
            f"Axes: {axes}\n"
            f"PFT step: {step}\n\n"
            "No statistics are calculated yet."
        )

    def export_placeholder_report(self) -> None:
        parent = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.project_root / "results"),
        )
        if not parent:
            return

        out = Path(parent) / "pft_results_placeholder.txt"
        out.write_text(
            "PFT results / statistics placeholder\n"
            "No statistical calculations have been implemented yet.\n",
            encoding="utf-8",
        )
        QMessageBox.information(self, "Export complete", f"Placeholder report saved:\n{out}")


def make_results_widget(napari_viewer):
    return PFTResultsWidget(napari_viewer)
