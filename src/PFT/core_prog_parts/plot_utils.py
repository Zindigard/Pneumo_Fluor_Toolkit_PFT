"""Provide command-line and programmatic utilities for plot utils."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from PFT.core_prog_parts.io import CziMeta

"""Shared plotting and savefig helpers used by visualization scripts."""


def add_scalebar(ax, meta: CziMeta | None, length_um: float = 5.0) -> None:
    """Add a white micron scalebar when pixel size metadata is available.

    Args:
        ax (Any): Matplotlib axes object on which graphical elements are drawn.
        meta (CziMeta | None): Value specifying meta for the operation.
        length_um (float): Numerical value controlling length um. Defaults to ``5.0``.

    Example:
        >>> add_scalebar(ax=..., meta=...)
    """
    if meta is None:
        return
    sx = meta.pixel_size_um_x
    if sx is None or sx <= 0:
        return
    length_px = length_um / sx
    fontprops = fm.FontProperties(size=10)
    bar = AnchoredSizeBar(
        ax.transData,
        size=length_px,
        label=f"{length_um:g} µm",
        loc='lower right',
        pad=0.4,
        color='white',
        frameon=False,
        size_vertical=max(1.0, length_px * 0.02),
        fontproperties=fontprops,
    )
    ax.add_artist(bar)


def apply_axis_style(ax, *, facecolor: str = 'black', title: str | None = None, title_color: str = 'white', axis_off: bool = True) -> None:
    """Apply a consistent background, title and axis visibility to one axis.

    Args:
        ax (Any): Matplotlib axes object on which graphical elements are drawn.
        facecolor (str): Text value specifying facecolor. Defaults to ``'black'``.
        title (str | None): Title displayed on the generated figure or report section. ``None`` selects the function's default behavior.
        title_color (str): Text value specifying title color. Defaults to ``'white'``.
        axis_off (bool): Boolean flag controlling axis off. Defaults to ``True``.

    Example:
        >>> apply_axis_style(ax=...)
    """
    ax.set_facecolor(facecolor)
    if axis_off:
        ax.axis('off')
    if title:
        ax.set_title(title, color=title_color)


def finalize_figure(fig, out_path: Path, *, dpi: int = 200, tight: bool = True, facecolor: str | None = None, pad_inches: float = 0.05) -> None:
    """Save a figure with consistent directory creation and optional tight layout.

    Args:
        fig (Any): Matplotlib figure object containing the generated visualization.
        out_path (Path): Filesystem path associated with out.
        dpi (int): Resolution of a generated figure in dots per inch. Defaults to ``200``.
        tight (bool): Boolean flag controlling tight. Defaults to ``True``.
        facecolor (str | None): Text value specifying facecolor. ``None`` selects the function's default behavior.
        pad_inches (float): Numerical value controlling pad inches. Defaults to ``0.05``.

    Example:
        >>> finalize_figure(fig=..., out_path=Path("path/to/resource"))
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches, facecolor=facecolor or fig.get_facecolor())
    plt.close(fig)


def save_side_by_side(fig_title: str, out_path: Path, panels: list[tuple], *, facecolor: str = 'black', dpi: int = 200) -> None:
    """Save a simple side-by-side panel figure from image/title pairs.

    Args:
        fig_title (str): Text value specifying fig title.
        out_path (Path): Filesystem path associated with out.
        panels (list[tuple]): Value specifying panels for the operation.
        facecolor (str): Text value specifying facecolor. Defaults to ``'black'``.
        dpi (int): Resolution of a generated figure in dots per inch. Defaults to ``200``.

    Example:
        >>> save_side_by_side(
        ...     fig_title="fig_title",
        ...     out_path=Path("path/to/resource"),
        ...     panels=[],
        ... )
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), facecolor=facecolor)
    if len(panels) == 1:
        axes = [axes]
    for ax, (img, title, kwargs) in zip(axes, panels):
        apply_axis_style(ax, facecolor=facecolor, title=title)
        ax.imshow(img, **(kwargs or {}))
    if fig_title:
        fig.suptitle(fig_title, color='white' if facecolor == 'black' else 'black')
    finalize_figure(fig, out_path, dpi=dpi, facecolor=facecolor)
