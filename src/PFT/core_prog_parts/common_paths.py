"""Provide command-line and programmatic utilities for common paths."""

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
    """Return the repository root by walking upward from the given path.

    Args:
        start (Path | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = find_project_root()
    """
    p = (start or Path(__file__)).expanduser().resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
        if (parent / "src" / "PFT").exists() and (parent / "results").exists():
            return parent
        if (parent / ".git").exists() and (parent / "src" / "PFT").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def resolve_project_path(
    path: str | Path,
    project_root: Path | None = None,
) -> Path:
    """Resolve a path predictably against the PFT repository root.

    Absolute paths remain absolute. Relative paths are interpreted relative to
    the repository root, not relative to the terminal's current directory.
    The returned path is normalized even when it does not exist yet.

    Args:
        path (str | Path): Filesystem path to the required input or output resource.
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = resolve_project_path(path="path")
    """
    root = Path(project_root or find_project_root()).expanduser().resolve()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def project_relative_path(
    path: str | Path,
    project_root: Path | None = None,
) -> str | None:
    """Return a portable POSIX path relative to the repository when possible.

    Args:
        path (str | Path): Filesystem path to the required input or output resource.
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        str | None: Generated or resolved text value.

    Example:
        >>> result = project_relative_path(path="path")
    """
    root = Path(project_root or find_project_root()).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return None


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the same path.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = ensure_dir(path=Path("path/to/resource"))
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_dataset_name(dataset: str) -> str:
    """Map dataset aliases to dataset names.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = normalize_dataset_name(dataset="2d_time")
    """
    ds = dataset.strip().lower()
    return _DATASET_ALIASES.get(ds, ds)


def results_img_dir(project_root: Path | None = None) -> Path:
    """Return the main results/img directory.

    Args:
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = results_img_dir()
    """
    return ensure_dir((project_root or find_project_root()) / "results" / "img")


def filtered_img_root(project_root: Path | None = None) -> Path:
    """Return the results/img/filtered directory used by filtered 2D outputs.

    Args:
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = filtered_img_root()
    """
    return ensure_dir(results_img_dir(project_root) / "filtered")


def results_filters_dir(project_root: Path | None = None) -> Path:
    """Return the results/Filters directory used by FFT-based pipelines.

    Args:
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = results_filters_dir()
    """
    return ensure_dir((project_root or find_project_root()) / "results" / "Filters")


def dataset_img_dir(dataset: str, project_root: Path | None = None) -> Path:
    """Return the canonical results/img/<dataset> directory.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = dataset_img_dir(dataset="2d_time")
    """
    return results_img_dir(project_root) / normalize_dataset_name(dataset)
