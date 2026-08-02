#!/usr/bin/env python3
"""
Patch the current PFT Omnipose training implementation to use RAdam.

Run once from the PFT project root:

    python scripts/segmentation/patch_omnipose_radam.py

The script updates:
    src/PFT/core_prog_parts/segmentation/instance_segmentation_core.py

Training changes:
    SGD=False
        Selects RAdam instead of the default SGD optimizer.

    rescale=False
        Keeps the prepared microscopy image scale unchanged.

    save_each=False
        Prevents creation of a separate model file at every save interval.

A backup is created before the source file is modified.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


MARKER = "# PFT_OMNIPOSE_RADAM_POLICY"


def find_project_root(start: Path) -> Path:
    """Return the nearest parent containing the expected PFT project structure."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "src" / "PFT").is_dir()
            and (candidate / "scripts" / "segmentation").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find the PFT project root. "
        "Run this script from inside Pneumo_Fluor_Toolkit_PFT."
    )


def patch_instance_segmentation_core(target: Path) -> bool:
    """Insert the RAdam training policy before the Omnipose model.train call."""
    source = target.read_text(encoding="utf-8")

    if MARKER in source:
        print(f"Already patched: {target}")
        return False

    function_match = re.search(r"(?m)^def _train_omnipose\s*\(", source)
    if function_match is None:
        raise RuntimeError(
            f"Function _train_omnipose was not found in:\n{target}"
        )

    function_start = function_match.start()

    next_function_match = re.search(
        r"(?m)^def\s+\w+\s*\(",
        source[function_match.end():],
    )
    function_end = (
        function_match.end() + next_function_match.start()
        if next_function_match is not None
        else len(source)
    )

    function_text = source[function_start:function_end]

    call_match = re.search(
        r"(?m)^(?P<indent>[ \t]+)"
        r"result\s*=\s*model\.train\(\*\*supported_train_values\)\s*$",
        function_text,
    )
    if call_match is None:
        raise RuntimeError(
            "Could not find the expected training call:\n"
            "    result = model.train(**supported_train_values)\n"
            "The current source file requires manual inspection."
        )

    indent = call_match.group("indent")

    insertion = (
        f"{indent}{MARKER}\n"
        f"{indent}from inspect import signature as _signature\n"
        f"\n"
        f"{indent}_train_parameters = _signature(model.train).parameters\n"
        f"{indent}_omnipose_training_overrides = {{\n"
        f'{indent}    "SGD": False,       # Use RAdam.\n'
        f'{indent}    "rescale": False,   # Preserve prepared image scale.\n'
        f'{indent}    "save_each": False, # Reuse the same checkpoint path.\n'
        f"{indent}}}\n"
        f"{indent}for _name, _value in _omnipose_training_overrides.items():\n"
        f"{indent}    if _name in _train_parameters:\n"
        f"{indent}        supported_train_values[_name] = _value\n"
        f"\n"
        f'{indent}print("Omnipose optimizer:       RAdam (SGD=False)")\n'
        f'{indent}print("Omnipose training rescale: disabled")\n'
        f'{indent}print("Omnipose save_each:        disabled")\n'
        f"\n"
    )

    insertion_index = function_start + call_match.start()
    patched = source[:insertion_index] + insertion + source[insertion_index:]

    compile(patched, str(target), "exec")

    backup = target.with_name(target.stem + ".before_omnipose_radam.py")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backup created: {backup}")
    else:
        print(f"Backup already exists: {backup}")

    target.write_text(patched, encoding="utf-8")
    print(f"Updated: {target}")
    return True


def main() -> int:
    """Apply the Omnipose RAdam patch to the active PFT project."""
    project_root = find_project_root(Path.cwd().resolve())

    target = (
        project_root
        / "src"
        / "PFT"
        / "core_prog_parts"
        / "segmentation"
        / "instance_segmentation_core.py"
    )

    if not target.is_file():
        raise FileNotFoundError(f"Target file does not exist:\n{target}")

    changed = patch_instance_segmentation_core(target)

    if changed:
        print("Result: Omnipose RAdam policy inserted successfully.")
    else:
        print("Result: no change was required.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
