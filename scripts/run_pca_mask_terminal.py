from __future__ import annotations

import sys
from pathlib import Path


_THIS = Path(__file__).resolve()
for candidate in (_THIS.parent, *_THIS.parents):
    src = candidate / "src"
    if (src / "PFT").exists():
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        break

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.pca_maskt_core import (
    PCAMaskAlignmentConfig,
    discover_prediction_runs,
    list_mask_samples,
    run_pca_alignment_2d_time,
    run_pca_alignment_2d_wga_dapi,
    run_pca_alignment_3d,
)


"""Interactive terminal launcher for PCA-based instance-mask alignment."""


def ask_choice(title: str, options: list[str], default: int = 0) -> str:
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        suffix = " (default)" if index - 1 == default else ""
        print(f"  {index}) {option}{suffix}")

    while True:
        value = input(
            f"Choose number [1-{len(options)}] or Enter for default: "
        ).strip()
        if not value:
            return options[default]
        try:
            selected = int(value) - 1
        except ValueError:
            selected = -1
        if 0 <= selected < len(options):
            return options[selected]
        print("Invalid choice.")


def ask_yes_no(text: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{text} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "1", "true"}:
            return True
        if value in {"n", "no", "0", "false"}:
            return False
        print("Please answer yes or no.")


def ask_int(text: str, default: int, minimum: int | None = None) -> int:
    while True:
        value = input(f"{text} [{default}]: ").strip()
        try:
            result = default if not value else int(value)
        except ValueError:
            print("Please enter an integer.")
            continue
        if minimum is not None and result < minimum:
            print(f"The value must be at least {minimum}.")
            continue
        return result


def ask_float(text: str, default: float, minimum: float | None = None) -> float:
    while True:
        value = input(f"{text} [{default}]: ").strip()
        try:
            result = default if not value else float(value)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum is not None and result < minimum:
            print(f"The value must be at least {minimum}.")
            continue
        return result


def ask_folder_name(text: str, default: str) -> str:
    while True:
        value = input(f"{text} [{default}]: ").strip() or default
        path = Path(value)
        if not path.is_absolute() and len(path.parts) == 1 and value not in {".", ".."}:
            return value
        print("Enter one folder name without path separators.")


def detect_project_root() -> Path:
    try:
        detected = find_project_root(Path(__file__).resolve())
    except TypeError:
        detected = find_project_root()

    print(f"\nDetected project root:\n  {detected}")
    if ask_yes_no("Use this project root?", True):
        return Path(detected).resolve()

    while True:
        value = input("Enter the Pneumo_Fluor_Toolkit_PFT project root: ").strip().strip('"')
        root = Path(value).expanduser().resolve()
        if (root / "src" / "PFT").exists() and (root / "results").exists():
            return root
        print("The selected folder does not contain src/PFT and results.")


def choose_method() -> str:
    displayed = ask_choice(
        "Choose segmentation method:",
        ["Cellpose", "Omnipose", "StarDist"],
        default=0,
    )
    return displayed.lower()


def choose_dataset() -> str:
    displayed = ask_choice(
        "Choose dataset:",
        ["2d_time", "2d_wga_dapi", "3d"],
        default=0,
    )
    return displayed


def choose_prediction_run(
    project_root: Path,
    method: str,
    dataset: str,
) -> Path:
    runs = discover_prediction_runs(project_root, method, dataset)

    if runs:
        print("\nAvailable prediction run folders:")
        for index, path in enumerate(runs, start=1):
            print(f"  {index}) {path}")
        print("  M) enter a folder manually")

        while True:
            value = input("Choose run folder [1]: ").strip().lower()
            if not value:
                return runs[0]
            if value in {"m", "manual"}:
                break
            try:
                selected = int(value) - 1
            except ValueError:
                selected = -1
            if 0 <= selected < len(runs):
                return runs[selected]
            print("Invalid choice.")
    else:
        print(
            "\nNo prediction run folder was discovered automatically. "
            "A manual path is required."
        )

    while True:
        value = input(
            "Prediction run folder containing sample/labels.tif or "
            "sample/labels.ome.zarr: "
        ).strip().strip('"')
        path = Path(value).expanduser().resolve()
        if list_mask_samples(path):
            return path
        print("No valid sample label masks were found in that folder.")


def choose_samples(prediction_run_dir: Path) -> tuple[str, ...] | None:
    samples = list_mask_samples(prediction_run_dir)
    if not samples:
        raise RuntimeError(f"No mask samples found in {prediction_run_dir}")

    print("\nAvailable mask samples:")
    for index, (sample_name, _sample_dir, mask_path) in enumerate(samples, start=1):
        print(f"  {index}) {sample_name} [{mask_path.name}]")
    print("  A) all samples")
    print("  C) choose several samples")

    while True:
        value = input("Choose one number, C, or A [A]: ").strip().lower()
        if value in {"", "a", "all"}:
            return None
        if value in {"c", "choose"}:
            selected_text = input(
                "Enter sample numbers separated by commas, for example 1,3,5: "
            ).strip()
            try:
                indices = sorted(
                    {int(part.strip()) - 1 for part in selected_text.split(",")}
                )
            except ValueError:
                print("Invalid sample-number list.")
                continue
            if indices and all(0 <= index < len(samples) for index in indices):
                return tuple(samples[index][0] for index in indices)
            print("One or more sample numbers are invalid.")
            continue
        try:
            selected = int(value) - 1
        except ValueError:
            selected = -1
        if 0 <= selected < len(samples):
            return (samples[selected][0],)
        print("Invalid choice.")


def build_config_interactive() -> PCAMaskAlignmentConfig:
    project_root = detect_project_root()

    # Required order: segmentation method first, then dataset.
    method = choose_method()
    dataset = choose_dataset()
    prediction_run_dir = choose_prediction_run(project_root, method, dataset)
    sample_names = choose_samples(prediction_run_dir)

    output_subfolder = ask_folder_name(
        "Output subfolder created inside each original sample-mask folder",
        "pca_aligned",
    )
    min_object_pixels = ask_int(
        "Minimum object area for PCA, in pixels",
        5,
        minimum=1,
    )
    min_anisotropy_ratio = ask_float(
        "Minimum major/minor PCA variance ratio for rotation",
        1.05,
        minimum=1.0,
    )
    crop_padding = ask_int(
        "Padding around each aligned cell crop, in pixels",
        4,
        minimum=0,
    )

    small_display = ask_choice(
        "How should objects smaller than the minimum area be handled?",
        ["Keep unchanged", "Exclude from aligned output"],
        default=0,
    )
    small_object_policy = "keep" if small_display == "Keep unchanged" else "exclude"

    collision_display = ask_choice(
        "How should rotated cells that overlap in the combined mask be handled?",
        [
            "Keep larger cells first and fill only empty pixels",
            "Allow later labels to overwrite earlier labels",
        ],
        default=0,
    )
    collision_policy = (
        "keep_larger"
        if collision_display.startswith("Keep larger")
        else "overwrite"
    )

    save_combined_mask = ask_yes_no(
        "Save a combined aligned instance-label mask?",
        True,
    )
    save_individual_cells = ask_yes_no(
        "Save every aligned cell as an individual TIFF crop?",
        True,
    )
    while not save_combined_mask and not save_individual_cells:
        print("At least one output type must be selected.")
        save_combined_mask = ask_yes_no(
            "Save a combined aligned instance-label mask?",
            True,
        )
        save_individual_cells = ask_yes_no(
            "Save every aligned cell as an individual TIFF crop?",
            True,
        )

    save_omezarr = False
    if save_combined_mask:
        save_omezarr = ask_yes_no(
            "Also save the combined mask as OME-Zarr?",
            True,
        )

    overwrite = ask_yes_no(
        "Overwrite existing PCA-alignment output folders?",
        False,
    )

    return PCAMaskAlignmentConfig(
        project_root=project_root,
        method=method,
        dataset=dataset,
        prediction_run_dir=prediction_run_dir,
        sample_names=sample_names,
        output_subfolder=output_subfolder,
        min_object_pixels=min_object_pixels,
        min_anisotropy_ratio=min_anisotropy_ratio,
        crop_padding=crop_padding,
        small_object_policy=small_object_policy,
        collision_policy=collision_policy,
        save_combined_mask=save_combined_mask,
        save_individual_cells=save_individual_cells,
        save_omezarr=save_omezarr,
        overwrite=overwrite,
    )


def print_config(cfg: PCAMaskAlignmentConfig) -> None:
    selected = "all" if cfg.sample_names is None else ", ".join(cfg.sample_names)
    print("\n=== PCA MASK ALIGNMENT CONFIGURATION ===")
    print(f"Project root:          {cfg.project_root}")
    print(f"Segmentation method:   {cfg.method}")
    print(f"Dataset:               {cfg.dataset}")
    print(f"Prediction run folder: {cfg.prediction_run_dir}")
    print(f"Samples:               {selected}")
    print(f"Output subfolder:      {cfg.output_subfolder}")
    print(f"Minimum object area:   {cfg.min_object_pixels} pixels")
    print(f"Minimum anisotropy:    {cfg.min_anisotropy_ratio}")
    print(f"Crop padding:          {cfg.crop_padding} pixels")
    print(f"Small-object policy:   {cfg.small_object_policy}")
    print(f"Collision policy:      {cfg.collision_policy}")
    print(f"Save combined mask:    {cfg.save_combined_mask}")
    print(f"Save individual cells: {cfg.save_individual_cells}")
    print(f"Save OME-Zarr:         {cfg.save_omezarr}")
    print(f"Overwrite:             {cfg.overwrite}")


def main() -> int:
    cfg = build_config_interactive()
    print_config(cfg)
    if not ask_yes_no("Run PCA alignment with these settings?", True):
        print("Cancelled.")
        return 0

    if cfg.dataset == "2d_time":
        outputs = run_pca_alignment_2d_time(cfg)
    elif cfg.dataset == "2d_wga_dapi":
        outputs = run_pca_alignment_2d_wga_dapi(cfg)
    elif cfg.dataset == "3d":
        outputs = run_pca_alignment_3d(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    print("\nPCA alignment finished. Output folders:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
