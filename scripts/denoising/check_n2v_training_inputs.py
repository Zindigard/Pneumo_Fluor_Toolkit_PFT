"""
Validate exactly what enters one two-dimensional Noise2Void model.

The checker confirms that model inputs are original level-0 OME-Zarr images
below ``results/img`` rather than normalized TIFFs, previews, masks, or previous
N2V outputs.  It verifies axes, selected channels, data type, finite sampled
values, metadata identity hashes, and, when training artifacts already exist,
the image-level train/validation manifest.

No model is trained and no image is modified.  Reports are written to
``results/N2V/reports`` as TXT, CSV, and JSON.

Examples
--------
Interactive model selection::

    python scripts/denoising/check_n2v_training_inputs.py

Check inputs for the joint DAPI+WGA model::

    python scripts/denoising/check_n2v_training_inputs.py --model joint
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_FILE = Path(__file__).resolve()


def find_project_root() -> Path:
    """Locate the repository root containing ``scripts`` and ``src/PFT``."""

    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT repository root.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PFT.core_prog_parts.denoising.n2v_workflow import (  # noqa: E402
    MODEL_SPECS,
    N2VModelSpec,
    choose_model_key_interactive,
    discover_original_samples,
    get_model_spec,
    level_zero_path,
    models_root,
    n2v_results_root,
    source_array_properties,
    utc_now_iso,
)


@dataclass(frozen=True)
class InputRecord:
    """Store validation results for one original OME-Zarr input."""

    status: str
    model_key: str
    dataset: str
    sample: str
    zarr_path: str
    axes: str
    shape: str
    dtype: str
    channels_requested: str
    sampled_minimum: float | None
    sampled_maximum: float | None
    sampled_mean: float | None
    input_normalization: str
    source_attrs_sha256: str
    issues: str


def _axis_size(axes: str, shape: Sequence[int], axis: str) -> int:
    """Return an axis length, treating an absent channel/time axis as one."""

    return int(shape[axes.index(axis)]) if axis in axes else 1


def sample_selected_values(zarr_path: Path, spec: N2VModelSpec, max_side: int = 256) -> np.ndarray:
    """Read a bounded value sample from the selected source channels.

    Non-spatial dimensions are fixed at index zero except for the requested
    channel.  Y and X are sampled with a stride so this check does not load the
    complete microscopy image into memory.
    """

    try:
        import zarr
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError("The training-input checker requires zarr.") from exc

    group = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(group.attrs)
    array = group[level_zero_path(attrs)]
    properties = source_array_properties(zarr_path)
    axes = str(properties["axes"])
    shape = tuple(int(value) for value in properties["shape"])
    y_step = max(1, int(np.ceil(_axis_size(axes, shape, "y") / max_side)))
    x_step = max(1, int(np.ceil(_axis_size(axes, shape, "x") / max_side)))

    selected: list[np.ndarray] = []
    for channel in spec.channels:
        index: list[Any] = []
        for axis, size in zip(axes, shape):
            if axis == "y":
                index.append(slice(None, None, y_step))
            elif axis == "x":
                index.append(slice(None, None, x_step))
            elif axis == "c":
                if not 0 <= int(channel) < int(size):
                    raise IndexError(f"Requested channel {channel} outside 0..{int(size) - 1}.")
                index.append(int(channel))
            else:
                index.append(0)
        selected.append(np.asarray(array[tuple(index)], dtype=np.float64).ravel())
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.float64)


def validate_one_input(zarr_path: Path, spec: N2VModelSpec) -> InputRecord:
    """Validate canonical location, metadata, channels, and sampled values."""

    failures: list[str] = []
    warnings: list[str] = []
    properties: dict[str, Any] = {}
    sampled = np.empty(0, dtype=np.float64)
    canonical_root = (PROJECT_ROOT / "results" / "img" / spec.dataset).resolve()

    try:
        resolved = zarr_path.resolve()
        if resolved.name != "image.ome.zarr":
            failures.append("input is not named image.ome.zarr")
        if canonical_root not in resolved.parents:
            failures.append(f"input is outside canonical original root {canonical_root}")
        properties = source_array_properties(resolved)
        axes = str(properties["axes"])
        shape = tuple(int(value) for value in properties["shape"])
        if "y" not in axes or "x" not in axes:
            failures.append(f"unsupported axes without YX: {axes}")
        channel_size = _axis_size(axes, shape, "c")
        invalid_channels = [channel for channel in spec.channels if channel < 0 or channel >= channel_size]
        if invalid_channels:
            failures.append(f"requested channels outside 0..{channel_size - 1}: {invalid_channels}")

        attrs = properties.get("attrs", {})
        if isinstance(attrs, Mapping):
            if attrs.get("pft_pipeline_stage") == "n2v_denoising":
                failures.append("input is an N2V-derived image, not an original image")
            if any(str(key).startswith("pft_n2v_") for key in attrs):
                failures.append("input contains N2V provenance attributes")
        if "norm" in str(resolved).lower() or "preview" in str(resolved).lower():
            failures.append("path name suggests a normalized or preview product")

        sampled = sample_selected_values(resolved, spec)
        if sampled.size == 0:
            failures.append("no numeric source values could be sampled")
        elif not np.all(np.isfinite(sampled)):
            failures.append("sampled source values contain NaN or infinity")
        elif np.min(sampled) >= 0.0 and np.max(sampled) <= 1.0 and np.issubdtype(np.dtype(properties["dtype"]), np.floating):
            warnings.append(
                "floating-point values are confined to [0,1]; verify that the original acquisition was not normalized upstream"
            )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    issues = "; ".join([*(f"FAIL: {item}" for item in failures), *(f"WARN: {item}" for item in warnings)])
    return InputRecord(
        status=status,
        model_key=spec.key,
        dataset=spec.dataset,
        sample=zarr_path.parent.name,
        zarr_path=str(zarr_path.resolve()),
        axes=str(properties.get("axes", "")),
        shape=str(tuple(properties.get("shape", []))),
        dtype=str(properties.get("dtype", "")),
        channels_requested=",".join(str(value) for value in spec.channels),
        sampled_minimum=float(np.min(sampled)) if sampled.size else None,
        sampled_maximum=float(np.max(sampled)) if sampled.size else None,
        sampled_mean=float(np.mean(sampled)) if sampled.size else None,
        input_normalization="none",
        source_attrs_sha256=str(properties.get("attrs_sha256", "")),
        issues=issues,
    )


def check_training_manifest(spec: N2VModelSpec, discovered: Sequence[Path]) -> list[str]:
    """Check an existing image-level split manifest for completeness and overlap."""

    issues: list[str] = []
    manifest = models_root(PROJECT_ROOT) / spec.model_name / "pft_training" / "sample_split.json"
    if not manifest.exists():
        return ["WARN: no sample_split.json exists yet; run training to create the final split manifest"]
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("manifest root is not a list")
        train = {str(row.get("zarr_path", "")) for row in rows if row.get("partition") == "train"}
        validation = {str(row.get("zarr_path", "")) for row in rows if row.get("partition") == "validation"}
        if train.intersection(validation):
            issues.append("FAIL: train and validation partitions overlap")
        if not train:
            issues.append("FAIL: manifest contains no training images")
        if not validation:
            issues.append("FAIL: manifest contains no validation images")

        def resolve_manifest_path(value: str) -> Path:
            """Resolve one manifest path relative to the repository when needed."""

            path = Path(value)
            return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

        manifest_paths = {resolve_manifest_path(value) for value in train.union(validation)}
        discovered_paths = {Path(path).resolve() for path in discovered}
        missing = manifest_paths - discovered_paths
        unassigned = discovered_paths - manifest_paths
        if missing:
            issues.append(f"FAIL: {len(missing)} manifest paths are not current original inputs")
        if unassigned:
            issues.append(f"WARN: {len(unassigned)} current original inputs are absent from the saved split")
    except Exception as exc:
        issues.append(f"FAIL: cannot read training manifest: {type(exc).__name__}: {exc}")
    return issues


def save_report(records: Sequence[InputRecord], spec: N2VModelSpec, manifest_issues: Sequence[str]) -> Path:
    """Write TXT, CSV, and JSON reports for the selected model inputs."""

    report_dir = n2v_results_root(PROJECT_ROOT) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"n2v_training_inputs_{spec.key}"
    rows = [asdict(record) for record in records]

    with (report_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / f"{stem}.json").write_text(
        json.dumps({"records": rows, "manifest_issues": list(manifest_issues)}, indent=2),
        encoding="utf-8",
    )

    passed = sum(record.status == "PASS" for record in records)
    warned = sum(record.status == "WARN" for record in records)
    failed = sum(record.status == "FAIL" for record in records)
    manifest_failed = any(issue.startswith("FAIL:") for issue in manifest_issues)
    overall = "FAIL" if failed or manifest_failed else ("WARN" if warned or manifest_issues else "PASS")
    lines = [
        "PFT 2D N2V training-input readiness report",
        "=" * 72,
        f"Generated (UTC): {utc_now_iso()}",
        f"Overall status: {overall}",
        f"Model key: {spec.key}",
        f"Model name: {spec.model_name}",
        f"Dataset: {spec.dataset}",
        f"Canonical source root: {PROJECT_ROOT / 'results' / 'img' / spec.dataset}",
        f"Channels supplied to model: {list(spec.channels)}",
        "Input normalization: none",
        f"Images discovered: {len(records)}",
        f"PASS: {passed}",
        f"WARN: {warned}",
        f"FAIL: {failed}",
        "",
        "Manifest checks",
        "---------------",
        *(manifest_issues or ["PASS: image-level split manifest is complete and disjoint"]),
        "",
        "Per-image checks",
        "----------------",
    ]
    for record in records:
        lines.append(
            f"{record.status} | {record.dataset} | {record.sample} | axes={record.axes} | "
            f"shape={record.shape} | dtype={record.dtype}"
            + (f" | {record.issues}" if record.issues else "")
        )
    txt_path = report_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for model-specific input checks."""

    parser = argparse.ArgumentParser(
        description="Check the original OME-Zarr images and channels supplied to one 2D N2V model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), help="Model input set to validate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate all canonical inputs and return nonzero only for blocking failures."""

    args = build_parser().parse_args(argv)
    model_key = args.model or choose_model_key_interactive("Choose the N2V model input set to check")
    spec = get_model_spec(model_key)
    samples = discover_original_samples(spec, PROJECT_ROOT)
    if not samples:
        raise SystemExit(
            f"No original image.ome.zarr files found below {PROJECT_ROOT / 'results' / 'img' / spec.dataset}."
        )
    records = [validate_one_input(path, spec) for path in samples]
    manifest_issues = check_training_manifest(spec, samples)
    report = save_report(records, spec, manifest_issues)
    print(f"Training-input report: {report}")
    has_failure = any(record.status == "FAIL" for record in records) or any(
        issue.startswith("FAIL:") for issue in manifest_issues
    )
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
