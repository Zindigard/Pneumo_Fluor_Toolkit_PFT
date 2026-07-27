from __future__ import annotations

from pathlib import Path

"""Shared path and project-directory helpers used across processing pipelines."""

_DATASET_ALIASES = {
    "2d_time": "2d_time",
    "2d_wga_dapi": "2d_wga_dapi",
    "2d_dpa_wagi": "2d_wga_dapi",
    "2d_dpa_wga": "2d_wga_dapi",
    "2d_dpa_wagi ": "2d_wga_dapi",
}


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward from the given path."""
    p = (start or Path(__file__)).resolve()
    for parent in [p, *p.parents]:
        if (parent / 'pyproject.toml').exists() and (parent / 'src').exists():
            return parent
        if (parent / 'src' / 'PFT').exists() and (parent / 'results').exists():
            return parent
        if (parent / '.git').exists():
            return parent
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_dataset_name(dataset: str) -> str:
    """Map dataset aliases to dataset names."""
    ds = dataset.strip().lower()
    return _DATASET_ALIASES.get(ds, ds)


def results_img_dir(project_root: Path | None = None) -> Path:
    """Return the main results/img directory."""
    return ensure_dir((project_root or find_project_root()) / 'results' / 'img')


def filtered_img_root(project_root: Path | None = None) -> Path:
    """Return the results/img/filtered directory used by filtered 2D outputs."""
    return ensure_dir(results_img_dir(project_root) / 'filtered')


def results_filters_dir(project_root: Path | None = None) -> Path:
    """Return the results/Filters directory used by FFT-based pipelines."""
    return ensure_dir((project_root or find_project_root()) / 'results' / 'Filters')


def dataset_img_dir(dataset: str, project_root: Path | None = None) -> Path:
    """Return the canonical results/img/<dataset> directory."""
    return results_img_dir(project_root) / normalize_dataset_name(dataset)
