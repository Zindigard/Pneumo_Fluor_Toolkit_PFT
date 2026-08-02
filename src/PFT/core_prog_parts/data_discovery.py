"""
Resolve raw microscopy-data directories without assuming a fixed drive letter.
The module recognizes standard PFT dataset folder names, searches parent
folders recursively, supports graphical or terminal-based folder selection,
and groups discovered CZI files by their relative experiment directories.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import os
import sys


DATASET_FOLDER_NAMES: dict[str, tuple[str, ...]] = {
    "2d_time": (
        "2d_data_time",
        "2D_data_time",
        "2d_time",
        "2D_time",
    ),
    "2d_wga_dapi": (
        "2d_data_WGA_DAPI",
        "2D_data_WGA_DAPI",
        "2d_data_WGA_DAPI_DNA",
        "2D_data_WGA_DAPI_DNA",
        "2d_wga_dapi",
        "WGA_DAPI",
    ),
    "3d": (
        "3d",
        "3D",
        "3d_data",
        "3D_data",
    ),
}

_SKIP_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "results",
    ".venv",
    "venv",
}


def _normalized_name(value: str) -> str:
    """Normalize a folder name to lowercase alphanumeric characters for robust alias matching.

    Args:
        value (str): Value to validate, transform, store, or forward.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _normalized_name(value="value")
    """
    return "".join(ch.lower() for ch in value if ch.isalnum())


def dataset_folder_aliases(dataset: str) -> tuple[str, ...]:
    """Return accepted folder-name aliases for a canonical PFT dataset identifier.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        tuple[str, ...]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = dataset_folder_aliases(dataset="2d_time")
    """
    try:
        return DATASET_FOLDER_NAMES[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset!r}") from exc


def _is_czi_file(path: Path) -> bool:
    """Return whether a path is a regular file with a case-insensitive ``.czi`` suffix.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _is_czi_file(path=Path("path/to/resource"))
    """
    return path.is_file() and path.suffix.lower() == ".czi"


def contains_czi(folder: str | Path, *, recursive: bool = True) -> bool:
    """Determine whether a folder contains at least one CZI file.

    Recursive searches skip repository, environment, cache, and results folders.

    Args:
        folder (str | Path): Filesystem path used for folder.
        recursive (bool): Boolean flag controlling recursive. Defaults to ``True``.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = contains_czi(folder="folder")
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return False

    if not recursive:
        return any(_is_czi_file(p) for p in folder.iterdir())

    for current, directories, files in os.walk(folder):
        directories[:] = [d for d in directories if d not in _SKIP_DIRECTORY_NAMES]
        if any(Path(name).suffix.lower() == ".czi" for name in files):
            return True
    return False


def _name_matches_dataset(folder: Path, dataset: str) -> bool:
    """Return whether a directory name matches one of the aliases for a dataset.

    Args:
        folder (Path): Filesystem path used for folder.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _name_matches_dataset(folder=Path("path/to/resource"), dataset="2d_time")
    """
    aliases = {_normalized_name(name) for name in dataset_folder_aliases(dataset)}
    return _normalized_name(folder.name) in aliases


def is_dataset_directory(folder: str | Path, dataset: str) -> bool:
    """Determine whether a directory can serve directly as a dataset root.

    A non-standard directory is accepted when it contains CZI files directly. A
    recognized dataset directory may contain CZI files in nested experiment folders.

    Args:
        folder (str | Path): Filesystem path used for folder.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = is_dataset_directory(folder="folder", dataset="2d_time")
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return False

    # Non-standard directories are accepted when CZI files are directly inside
    # them. Conventionally named directories may contain nested experiment
    # folders and are therefore searched recursively.
    return contains_czi(folder, recursive=False) or (
        _name_matches_dataset(folder, dataset) and contains_czi(folder, recursive=True)
    )


def find_dataset_directories(search_root: str | Path, dataset: str) -> list[Path]:
    """Search recursively for recognized dataset directories containing CZI files.

    Returns unique absolute paths in deterministic order.

    Args:
        search_root (str | Path): Directory used for search.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        NotADirectoryError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_dataset_directories(search_root=Path("path/to/resource"), dataset="2d_time")
    """
    root = Path(search_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Search location does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Search location is not a directory: {root}")

    aliases = {_normalized_name(name) for name in dataset_folder_aliases(dataset)}
    candidates: list[Path] = []

    def consider(path: Path) -> None:
        """Add a directory to the local candidate list when its name and contents match the dataset.

        Args:
            path (Path): Filesystem path to the required input or output resource.

        Example:
            >>> consider(path=Path("path/to/resource"))
        """
        if _normalized_name(path.name) in aliases and contains_czi(path, recursive=True):
            candidates.append(path.resolve())

    consider(root)
    for current, directories, _files in os.walk(root):
        directories[:] = [d for d in directories if d not in _SKIP_DIRECTORY_NAMES]
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if _normalized_name(directory) in aliases:
                consider(candidate)

    # Preserve deterministic ordering and remove duplicates caused by nested walks.
    unique = sorted(set(candidates), key=lambda p: (len(p.parts), str(p).lower()))
    return unique


def choose_path_from_terminal(paths: Iterable[Path], *, prompt: str) -> Path:
    """Present candidate paths in the terminal and return the selected path.

    Args:
        paths (Iterable[Path]): Filesystem path used for paths.
        prompt (str): Text value specifying prompt.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = choose_path_from_terminal(paths=Path("path/to/resource"), prompt="prompt")
    """
    items = list(paths)
    if not items:
        raise ValueError("No paths were supplied for selection.")
    if len(items) == 1:
        return items[0]

    print(f"\n{prompt}")
    for index, path in enumerate(items):
        print(f"  [{index}] {path}")
    value = input("Select number: ").strip()
    index = int(value) if value else 0
    if not 0 <= index < len(items):
        raise ValueError(f"Selection {index} is outside 0..{len(items) - 1}")
    return items[index]


def select_directory_dialog(title: str) -> Path | None:
    """Open a native folder-selection dialog and return the selected path.

    Returns ``None`` when Tk is unavailable, the dialog fails, or selection is cancelled.

    Args:
        title (str): Title displayed on the generated figure or report section.

    Returns:
        Path | None: Resolved or generated filesystem path.

    Example:
        >>> result = select_directory_dialog(title="title")
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(title=title, mustexist=True)
        root.destroy()
        return Path(selected).expanduser() if selected else None
    except Exception:
        return None


def _request_search_root(dataset: str) -> Path:
    """Request a dataset directory or parent search root interactively.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        NotADirectoryError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _request_search_root(dataset="2d_time")
    """
    print("\nThe configured raw-data location could not be used.")
    print("Choose how to provide the raw-data location:")
    print("  [0] Select a folder in a file dialog")
    print("  [1] Enter a folder path in the terminal")
    choice = input("Select number (default 0): ").strip() or "0"

    selected: Path | None = None
    if choice == "0":
        selected = select_directory_dialog(
            f"Select the {dataset} dataset folder or a parent raw-data folder"
        )
        if selected is None:
            print("A graphical folder dialog is unavailable or was cancelled.")
    elif choice != "1":
        raise ValueError("Expected selection 0 or 1.")

    if selected is None:
        value = input("Enter the full folder path: ").strip().strip('"')
        if not value:
            raise FileNotFoundError("No raw-data location was supplied.")
        selected = Path(value).expanduser()

    if not selected.exists():
        raise FileNotFoundError(f"Selected location does not exist: {selected}")
    if not selected.is_dir():
        raise NotADirectoryError(f"Selected location is not a directory: {selected}")
    return selected


def resolve_dataset_directory(
    dataset: str,
    *,
    preferred: str | Path | None = None,
    search_root: str | Path | None = None,
    interactive: bool = True,
) -> Path:
    """Resolve the raw-data directory for a selected dataset.

    Resolution proceeds through a preferred path, an optional search root, and
    interactive selection. The returned path always contains discoverable CZI data.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        preferred (str | Path | None): Filesystem path used for preferred. ``None`` selects the function's default behavior.
        search_root (str | Path | None): Directory used for search. ``None`` selects the function's default behavior.
        interactive (bool): Boolean flag controlling interactive. Defaults to ``True``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_dataset_directory(dataset="2d_time")
    """
    if dataset not in DATASET_FOLDER_NAMES:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    attempted: list[Path] = []

    if preferred is not None:
        preferred_path = Path(preferred).expanduser()
        attempted.append(preferred_path)
        if preferred_path.is_dir():
            if is_dataset_directory(preferred_path, dataset):
                print(f"Using raw-data directory: {preferred_path.resolve()}")
                return preferred_path.resolve()
            matches = find_dataset_directories(preferred_path, dataset)
            if matches:
                selected = (
                    choose_path_from_terminal(matches, prompt="Multiple dataset directories found:")
                    if interactive
                    else matches[0]
                )
                print(f"Using discovered raw-data directory: {selected}")
                return selected

    if search_root is not None:
        root = Path(search_root).expanduser()
        attempted.append(root)
        matches = find_dataset_directories(root, dataset)
        if matches:
            selected = (
                choose_path_from_terminal(matches, prompt="Multiple dataset directories found:")
                if interactive
                else matches[0]
            )
            print(f"Using discovered raw-data directory: {selected}")
            return selected

    if not interactive:
        tried = ", ".join(str(p) for p in attempted) or "no location"
        aliases = ", ".join(dataset_folder_aliases(dataset))
        raise FileNotFoundError(
            f"Could not resolve raw data for {dataset}. Tried: {tried}. "
            f"Expected a directory containing CZI files or one named: {aliases}"
        )

    selected_root = _request_search_root(dataset)
    if is_dataset_directory(selected_root, dataset):
        print(f"Using selected raw-data directory: {selected_root.resolve()}")
        return selected_root.resolve()

    matches = find_dataset_directories(selected_root, dataset)
    if matches:
        selected = choose_path_from_terminal(
            matches,
            prompt="Dataset directories found below the selected location:",
        )
        print(f"Using discovered raw-data directory: {selected}")
        return selected

    aliases = ", ".join(dataset_folder_aliases(dataset))
    raise FileNotFoundError(
        f"No CZI files and no matching {dataset} directory were found below "
        f"{selected_root}. Recognized names: {aliases}"
    )


def group_czi_files_by_relative_parent(base_folder: str | Path) -> dict[str, list[Path]]:
    """Group recursively discovered CZI files by parent directory relative to a base folder.

    The dictionary keys are portable forward-slash experiment paths and the values
    are sorted lists of source CZI files.

    Args:
        base_folder (str | Path): Filesystem path used for base folder.

    Returns:
        dict[str, list[Path]]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = group_czi_files_by_relative_parent(base_folder="base_folder")
    """
    base = Path(base_folder).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"3D raw-data directory not found: {base}")

    files = sorted(
        (p for p in base.rglob("*") if _is_czi_file(p)),
        key=lambda p: str(p).lower(),
    )
    if not files:
        raise FileNotFoundError(f"No .czi files found recursively in: {base}")

    groups: dict[str, list[Path]] = {}
    for path in files:
        relative_parent = path.parent.relative_to(base)
        group_name = str(relative_parent).replace("\\", "/")
        if group_name in ("", "."):
            group_name = base.name
        groups.setdefault(group_name, []).append(path)
    return groups


def stdin_is_interactive() -> bool:
    """Return whether standard input supports interactive terminal prompts.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = stdin_is_interactive()
    """
    return bool(getattr(sys.stdin, "isatty", lambda: False)())
