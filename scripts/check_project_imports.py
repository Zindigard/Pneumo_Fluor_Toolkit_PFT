"""Validate the repository layout and all local PFT imports.

This script checks that ``scripts`` and ``src/PFT/core_prog_parts`` share the
same repository root. It then verifies every local ``PFT...`` import used by
the scripts and core modules without importing optional machine-learning or
microscopy dependencies.
"""

from __future__ import annotations
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))


import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportIssue:
    """Describe one unresolved local module or imported symbol."""

    file: Path
    target: str
    reason: str


def module_name_from_path(path: Path, src_dir: Path) -> str:
    """Convert a Python source path below ``src`` into its importable name."""
    relative = path.relative_to(src_dir)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    return ".".join(parts)[:-3]


def collect_top_level_symbols(path: Path) -> set[str]:
    """Return names defined or imported at module scope in one Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    symbols.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)

    return symbols


def build_module_index(src_dir: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """Index local modules and their top-level symbols below ``src``."""
    modules: dict[str, Path] = {}

    for path in src_dir.rglob("*.py"):
        modules[module_name_from_path(path, src_dir)] = path

    symbols = {
        module_name: collect_top_level_symbols(path)
        for module_name, path in modules.items()
    }
    return modules, symbols


def check_file_imports(
    path: Path,
    modules: dict[str, Path],
    symbols: dict[str, set[str]],
) -> list[ImportIssue]:
    """Check all absolute ``PFT`` imports found in one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    issues: list[ImportIssue] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                if not target.startswith("PFT"):
                    continue
                module_exists = target in modules or any(
                    name.startswith(f"{target}.") for name in modules
                )
                if not module_exists:
                    issues.append(ImportIssue(path, target, "module not found"))

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if not module_name or not module_name.startswith("PFT"):
                continue

            if module_name not in modules:
                issues.append(ImportIssue(path, module_name, "module not found"))
                continue

            for alias in node.names:
                if alias.name == "*":
                    continue
                submodule_name = f"{module_name}.{alias.name}"
                if (
                    alias.name not in symbols[module_name]
                    and submodule_name not in modules
                ):
                    issues.append(
                        ImportIssue(
                            path,
                            f"{module_name}.{alias.name}",
                            "symbol or submodule not found",
                        )
                    )

    return issues


def run_check(project_root: Path) -> tuple[list[str], list[ImportIssue]]:
    """Run repository, package, syntax, and local-import checks."""
    scripts_dir = project_root / "scripts"
    src_dir = project_root / "src"
    core_dir = src_dir / "PFT" / "core_prog_parts"

    messages = [
        f"Project root: {project_root}",
        f"Scripts directory: {scripts_dir}",
        f"Core directory: {core_dir}",
    ]

    if not scripts_dir.is_dir():
        return messages, [ImportIssue(scripts_dir, "scripts", "directory not found")]
    if not core_dir.is_dir():
        return messages, [
            ImportIssue(core_dir, "src/PFT/core_prog_parts", "directory not found")
        ]

    modules, symbols = build_module_index(src_dir)
    files = sorted(scripts_dir.rglob("*.py")) + sorted(core_dir.rglob("*.py"))
    issues: list[ImportIssue] = []

    for path in files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as error:
            issues.append(ImportIssue(path, "Python syntax", str(error)))
            continue
        issues.extend(check_file_imports(path, modules, symbols))

    messages.extend(
        [
            f"Python files checked: {len(files)}",
            f"Local PFT modules indexed: {len(modules)}",
        ]
    )
    return messages, issues


def write_report(
    output_path: Path,
    messages: list[str],
    issues: list[ImportIssue],
) -> None:
    """Write the import-check result to a plain-text report."""
    lines = ["PFT SCRIPT IMPORT CHECK", "=" * 23, *messages, ""]

    if issues:
        lines.append("STATUS: FAIL")
        for issue in issues:
            lines.append(
                f"- {issue.file}: {issue.target}: {issue.reason}"
            )
    else:
        lines.append("STATUS: PASS")
        lines.append("All local PFT import targets and imported symbols were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the import checker."""
    parser = argparse.ArgumentParser(
        description="Check the PFT repository layout and local Python imports."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PFT_PROJECT_ROOT / "results" / "import_check_report.txt",
        help="Text report path.",
    )
    return parser


def main() -> int:
    """Run the checks, print their results, and return a process status code."""
    args = build_parser().parse_args()
    messages, issues = run_check(_PFT_PROJECT_ROOT)
    write_report(args.output, messages, issues)

    for message in messages:
        print(message)

    if issues:
        print("STATUS: FAIL")
        for issue in issues:
            print(f"[FAIL] {issue.file}: {issue.target}: {issue.reason}")
        print(f"Report: {args.output}")
        return 1

    print("STATUS: PASS")
    print("All local PFT imports resolve correctly.")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
