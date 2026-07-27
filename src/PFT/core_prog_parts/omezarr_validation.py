"""
Validate OME-Zarr exports and create compact dataset-level summaries.

Each OME-Zarr image receives a detailed ``ome_zarr_validation.txt`` report that
contains exact level-0 pixel, dtype, shape, axis, scale, pyramid, channel, XML,
and complete ``CziMeta`` comparisons.

The module also scans per-image reports and writes one validation summary in
each dataset directory plus a global summary under ``results/img``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Iterable
import json
import numpy as np

from PFT.core_prog_parts.io import CziMeta
from PFT.core_prog_parts.omezarr_utils import (
    metadata_as_json_dict,
    prepare_array_and_axes,
    scale_vector_from_meta_um,
)


REPORT_FILENAME = "ome_zarr_validation.txt"
DATASET_SUMMARY_FILENAME = "ome_zarr_validation_summary.txt"
GLOBAL_SUMMARY_FILENAME = "ome_zarr_validation_global_summary.txt"


class OMEZarrValidationError(RuntimeError):
    """Indicate that one or more required OME-Zarr preservation checks failed."""


@dataclass(frozen=True)
class ValidationCheck:
    """Represent one named validation result with pass status and diagnostic detail."""
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    """Collect all checks and report paths produced for one OME-Zarr image."""
    zarr_path: Path
    report_path: Path
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        """Return ``True`` only when every validation check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        """Return all checks that failed in their original evaluation order."""
        return tuple(check for check in self.checks if not check.passed)

    @property
    def checks_passed(self) -> int:
        """Return the number of successful validation checks."""
        return sum(check.passed for check in self.checks)

    @property
    def checks_failed(self) -> int:
        """Return the number of failed validation checks."""
        return len(self.checks) - self.checks_passed


@dataclass(frozen=True)
class ValidationSummaryRow:
    """Represent one image in dataset-level and global validation summaries."""

    dataset: str
    sample: str
    status: str
    checks_passed: int
    checks_failed: int
    source_path: str
    zarr_path: str
    detail: str = ""


def _json_normalize(value: Any) -> Any:
    """Normalize Zarr attribute values into stable Python containers for comparison."""
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _short(value: Any, max_chars: int = 180) -> str:
    """Create a bounded-length textual representation for report diagnostics."""
    if isinstance(value, str) and len(value) > max_chars:
        return f"{value[:max_chars]}... <{len(value)} characters>"
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(value)
    return text if len(text) <= max_chars else f"{text[:max_chars]}..."


def _same_float_sequence(actual: Any, expected: Iterable[float]) -> bool:
    """Compare two numeric sequences using strict floating-point tolerances."""
    try:
        return bool(np.allclose(
            np.asarray(actual, dtype=float),
            np.asarray(list(expected), dtype=float),
            rtol=1e-9,
            atol=1e-12,
        ))
    except Exception:
        return False


def _dataset_entries(root: Any) -> tuple[list[dict[str, Any]], str]:
    """Extract multiscale dataset entries and concatenated axis names from OME-NGFF metadata."""
    multiscales = _json_normalize(root.attrs.get("multiscales", []))
    if not isinstance(multiscales, list) or not multiscales:
        return [], ""
    first = multiscales[0]
    if not isinstance(first, dict):
        return [], ""

    axes_entries = first.get("axes", [])
    axes = "".join(
        str(item.get("name", "")) if isinstance(item, dict) else str(item)
        for item in axes_entries
    ).lower()
    datasets = first.get("datasets", [])
    return (datasets if isinstance(datasets, list) else []), axes


def _iter_chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...] | None) -> Iterable[tuple[slice, ...]]:
    """Yield slice tuples that cover an array according to its chunk geometry."""
    if not shape:
        yield tuple()
        return
    if chunks is None or len(chunks) != len(shape):
        chunks = shape
    starts = [range(0, size, max(1, int(chunk))) for size, chunk in zip(shape, chunks)]
    for start_tuple in product(*starts):
        yield tuple(
            slice(start, min(start + max(1, int(chunk)), size))
            for start, size, chunk in zip(start_tuple, shape, chunks)
        )


def _hash_and_compare_exact(source: np.ndarray, zarr_array: Any) -> tuple[bool, str, str, str]:
    """Compare source and stored level-0 data chunk by chunk and compute SHA-256 digests."""
    source_hash = sha256()
    zarr_hash = sha256()
    exact = True
    mismatch_at = ""

    chunks_value = getattr(zarr_array, "chunks", None)
    chunks = tuple(int(v) for v in chunks_value) if chunks_value is not None else None

    for slice_tuple in _iter_chunk_slices(tuple(source.shape), chunks):
        source_chunk = np.ascontiguousarray(source[slice_tuple])
        zarr_chunk = np.ascontiguousarray(np.asarray(zarr_array[slice_tuple]))
        source_hash.update(memoryview(source_chunk).cast("B"))
        zarr_hash.update(memoryview(zarr_chunk).cast("B"))
        if exact and not np.array_equal(source_chunk, zarr_chunk, equal_nan=True):
            exact = False
            mismatch_at = repr(slice_tuple)

    detail = "all chunks are identical" if exact else f"first unequal chunk: {mismatch_at}"
    return exact, source_hash.hexdigest(), zarr_hash.hexdigest(), detail


def _extract_scale(dataset_entry: dict[str, Any]) -> list[float] | None:
    """Return the scale coordinate transformation from one OME-NGFF dataset entry."""
    transforms = dataset_entry.get("coordinateTransformations", [])
    if not isinstance(transforms, list):
        return None
    for transform in transforms:
        if isinstance(transform, dict) and transform.get("type") == "scale":
            scale = transform.get("scale")
            return list(scale) if isinstance(scale, list) else None
    return None


def _sidecar_xml_check(zarr_path: Path, meta: CziMeta) -> ValidationCheck:
    """Compare the metadata XML sidecar with the original CZI XML when a sidecar exists."""
    expected_xml = getattr(meta, "raw_xml", None)
    candidates = [zarr_path.parent / "metadata.xml", zarr_path.parent / "metadata_full.xml"]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        return ValidationCheck(
            "XML metadata sidecar",
            True,
            "not applicable in this output directory",
        )

    actual = existing.read_text(encoding="utf-8")
    if isinstance(expected_xml, str) and expected_xml.strip():
        passed = actual == expected_xml
        detail = f"exact match: {existing.name}" if passed else f"content differs: {existing.name}"
    else:
        passed = actual.strip() == "No XML metadata available."
        detail = f"absence marker verified: {existing.name}" if passed else f"unexpected XML content: {existing.name}"
    return ValidationCheck("XML metadata sidecar", passed, detail)


def validate_ome_zarr(
    zarr_path: str | Path,
    source_arr: np.ndarray,
    source_meta: CziMeta,
    *,
    report_path: str | Path | None = None,
    print_terminal: bool = True,
) -> ValidationResult:
    """Validate exact pixel data, axes, physical calibration, pyramid structure, and metadata.
    
    A detailed report is written beside the OME-Zarr directory. The function
    returns a ``ValidationResult`` rather than raising for failed checks.
    """
    zarr_path = Path(zarr_path).expanduser().resolve()
    report = Path(report_path) if report_path is not None else zarr_path.parent / REPORT_FILENAME
    report = report.expanduser().resolve()

    checks: list[ValidationCheck] = []
    checks.append(ValidationCheck("OME-Zarr directory exists", zarr_path.is_dir(), str(zarr_path)))
    if not zarr_path.is_dir():
        result = ValidationResult(zarr_path, report, tuple(checks))
        _write_report(result, source_meta, print_terminal=print_terminal)
        return result

    try:
        import zarr
    except Exception as exc:
        raise ImportError(
            "OME-Zarr validation requires zarr. Install project dependencies from pyproject.toml."
        ) from exc

    root = zarr.open_group(str(zarr_path), mode="r")
    prepared_arr, expected_axes = prepare_array_and_axes(source_arr, source_meta)
    datasets, actual_axes = _dataset_entries(root)

    checks.append(ValidationCheck(
        "OME-NGFF multiscales metadata",
        bool(datasets),
        f"dataset levels: {len(datasets)}" if datasets else "missing multiscales/datasets",
    ))
    checks.append(ValidationCheck(
        "Axis order",
        actual_axes == expected_axes,
        f"expected={expected_axes}, stored={actual_axes or '<missing>'}",
    ))

    level0_path = str(datasets[0].get("path", "0")) if datasets else "0"
    level0_exists = level0_path in root
    checks.append(ValidationCheck("Level-0 array exists", level0_exists, f"path={level0_path}"))

    if level0_exists:
        level0 = root[level0_path]
        shape_matches = tuple(level0.shape) == tuple(prepared_arr.shape)
        dtype_matches = np.dtype(level0.dtype) == prepared_arr.dtype
        checks.append(ValidationCheck(
            "Level-0 shape",
            shape_matches,
            f"expected={prepared_arr.shape}, stored={tuple(level0.shape)}",
        ))
        checks.append(ValidationCheck(
            "Level-0 dtype",
            dtype_matches,
            f"expected={prepared_arr.dtype}, stored={level0.dtype}",
        ))
        if shape_matches and dtype_matches:
            exact, source_digest, stored_digest, exact_detail = _hash_and_compare_exact(prepared_arr, level0)
            checks.append(ValidationCheck("Exact pixel identity", exact, exact_detail))
            checks.append(ValidationCheck(
                "Level-0 SHA-256",
                source_digest == stored_digest,
                f"source={source_digest}; stored={stored_digest}",
            ))
        else:
            checks.append(ValidationCheck("Exact pixel identity", False, "not evaluated because shape or dtype differs"))
            checks.append(ValidationCheck("Level-0 SHA-256", False, "not evaluated because shape or dtype differs"))

    expected_meta = _json_normalize(metadata_as_json_dict(source_meta))
    actual_meta = _json_normalize(root.attrs.get("pft_meta", {}))
    checks.append(ValidationCheck(
        "Complete pft_meta",
        actual_meta == expected_meta,
        "all fields match" if actual_meta == expected_meta else "one or more fields differ",
    ))

    for field in fields(source_meta):
        expected_value = expected_meta.get(field.name)
        actual_value = actual_meta.get(field.name) if isinstance(actual_meta, dict) else None
        passed = actual_value == expected_value
        checks.append(ValidationCheck(
            f"Metadata field: {field.name}",
            passed,
            "match" if passed else f"expected={_short(expected_value)}; stored={_short(actual_value)}",
        ))

    expected_source_path = str(getattr(source_meta, "source_path", ""))
    stored_source_path = str(root.attrs.get("source_path", ""))
    checks.append(ValidationCheck(
        "Root source_path",
        stored_source_path == expected_source_path,
        f"expected={expected_source_path}; stored={stored_source_path}",
    ))

    expected_channels = _json_normalize(getattr(source_meta, "channel_names", None))
    stored_channels = _json_normalize(root.attrs.get("channel_names", None))
    checks.append(ValidationCheck(
        "Root channel names",
        stored_channels == expected_channels,
        f"expected={_short(expected_channels)}; stored={_short(stored_channels)}",
    ))

    expected_scale = scale_vector_from_meta_um(expected_axes, source_meta)
    level0_scale = _extract_scale(datasets[0]) if datasets else None
    checks.append(ValidationCheck(
        "Level-0 physical scale",
        level0_scale is not None and _same_float_sequence(level0_scale, expected_scale),
        f"expected={expected_scale}; stored={level0_scale}",
    ))

    for level_index, entry in enumerate(datasets):
        path = str(entry.get("path", level_index))
        exists = path in root
        checks.append(ValidationCheck(f"Pyramid level {level_index} exists", exists, f"path={path}"))
        if not exists:
            continue
        array = root[path]
        checks.append(ValidationCheck(
            f"Pyramid level {level_index} dtype",
            np.dtype(array.dtype) == prepared_arr.dtype,
            f"expected={prepared_arr.dtype}; stored={array.dtype}",
        ))
        scale = _extract_scale(entry)
        checks.append(ValidationCheck(
            f"Pyramid level {level_index} scale metadata",
            scale is not None and len(scale) == len(expected_axes),
            f"scale={scale}",
        ))
        if level_index > 0 and level0_exists:
            level0_shape = tuple(root[level0_path].shape)
            level_shape = tuple(array.shape)
            geometry_ok = len(level_shape) == len(level0_shape)
            if geometry_ok:
                for axis_index, axis in enumerate(expected_axes):
                    if axis in "yx":
                        geometry_ok = geometry_ok and 1 <= level_shape[axis_index] <= level0_shape[axis_index]
                    else:
                        geometry_ok = geometry_ok and level_shape[axis_index] == level0_shape[axis_index]
            checks.append(ValidationCheck(
                f"Pyramid level {level_index} geometry",
                geometry_ok,
                f"level0={level0_shape}; level={level_shape}",
            ))

    checks.append(_sidecar_xml_check(zarr_path, source_meta))

    result = ValidationResult(zarr_path, report, tuple(checks))
    _write_report(result, source_meta, print_terminal=print_terminal)
    return result


def _format_report(result: ValidationResult, source_meta: CziMeta) -> str:
    """Format validation checks and source information as a human-readable text report."""
    passed_count = sum(check.passed for check in result.checks)
    failed_count = len(result.checks) - passed_count
    status = "PASS" if result.passed else "FAIL"
    lines = [
        "PFT OME-Zarr validation report",
        "=" * 80,
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Status: {status}",
        f"OME-Zarr: {result.zarr_path}",
        f"Source CZI: {getattr(source_meta, 'source_path', '')}",
        f"Checks passed: {passed_count}",
        f"Checks failed: {failed_count}",
        "",
        f"{'RESULT':<8} | CHECK | DETAIL",
        "-" * 80,
    ]
    for check in result.checks:
        marker = "PASS" if check.passed else "FAIL"
        lines.append(f"{marker:<8} | {check.name} | {check.detail}")
    lines.extend([
        "",
        "Interpretation:",
        "PASS means that level-0 pixels and all required metadata were preserved.",
        "FAIL means that the OME-Zarr output must not be used until the failed checks are resolved.",
        "",
    ])
    return "\n".join(lines)


def _write_report(
    result: ValidationResult,
    source_meta: CziMeta,
    *,
    print_terminal: bool,
) -> None:
    """Write a validation report to disk and optionally print the same report in the terminal."""
    text = _format_report(result, source_meta)
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(text, encoding="utf-8")
    if print_terminal:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"Validation {status}: {result.zarr_path} | "
            f"checks passed={result.checks_passed}, failed={result.checks_failed} | "
            f"report={result.report_path}"
        )



def make_validation_summary_row(
    *,
    dataset: str,
    sample: str,
    status: str,
    source_path: str | Path = "",
    zarr_path: str | Path = "",
    checks_passed: int = 0,
    checks_failed: int = 0,
    detail: str = "",
) -> ValidationSummaryRow:
    """Create a normalized summary row for successful, failed, or skipped work."""
    normalized_status = status.strip().upper()
    if normalized_status not in {"PASS", "FAIL", "ERROR", "SKIPPED"}:
        raise ValueError(f"Unsupported validation status: {status!r}")
    return ValidationSummaryRow(
        dataset=str(dataset),
        sample=str(sample),
        status=normalized_status,
        checks_passed=int(checks_passed),
        checks_failed=int(checks_failed),
        source_path=str(source_path),
        zarr_path=str(zarr_path),
        detail=str(detail),
    )


def summary_row_from_result(
    result: ValidationResult,
    source_meta: CziMeta,
    *,
    dataset: str,
    sample: str,
) -> ValidationSummaryRow:
    """Convert an in-memory validation result into one compact summary row."""
    return make_validation_summary_row(
        dataset=dataset,
        sample=sample,
        status="PASS" if result.passed else "FAIL",
        source_path=getattr(source_meta, "source_path", ""),
        zarr_path=result.zarr_path,
        checks_passed=result.checks_passed,
        checks_failed=result.checks_failed,
        detail="all required checks passed" if result.passed else "one or more checks failed",
    )


def _report_header_value(lines: list[str], label: str) -> str:
    """Extract a value from a ``Label: value`` line in a detailed report."""
    prefix = f"{label}:"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def summary_row_from_report(
    report_path: str | Path,
    *,
    results_img_root: str | Path,
) -> ValidationSummaryRow:
    """Parse one detailed report into a dataset/global summary row."""
    report = Path(report_path).expanduser().resolve()
    root = Path(results_img_root).expanduser().resolve()
    lines = report.read_text(encoding="utf-8").splitlines()

    try:
        relative_sample = report.parent.relative_to(root)
        dataset = relative_sample.parts[0] if relative_sample.parts else "unknown"
        sample_parts = relative_sample.parts[1:]
        sample = "/".join(sample_parts) if sample_parts else report.parent.name
    except ValueError:
        dataset = report.parent.parent.name or "unknown"
        sample = report.parent.name

    status = _report_header_value(lines, "Status").upper() or "ERROR"
    if status not in {"PASS", "FAIL", "ERROR", "SKIPPED"}:
        status = "ERROR"
    try:
        checks_passed = int(_report_header_value(lines, "Checks passed") or 0)
    except ValueError:
        checks_passed = 0
    try:
        checks_failed = int(_report_header_value(lines, "Checks failed") or 0)
    except ValueError:
        checks_failed = 0

    return make_validation_summary_row(
        dataset=dataset,
        sample=sample,
        status=status,
        source_path=_report_header_value(lines, "Source CZI"),
        zarr_path=_report_header_value(lines, "OME-Zarr"),
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        detail="detailed report available",
    )


def scan_validation_reports(results_img_root: str | Path) -> list[ValidationSummaryRow]:
    """Read every per-image validation report below a ``results/img`` directory."""
    root = Path(results_img_root).expanduser().resolve()
    if not root.is_dir():
        return []
    rows: list[ValidationSummaryRow] = []
    for report in sorted(root.rglob(REPORT_FILENAME), key=lambda path: str(path).lower()):
        try:
            rows.append(summary_row_from_report(report, results_img_root=root))
        except Exception as exc:
            relative = report.parent.relative_to(root) if report.parent.is_relative_to(root) else Path(report.parent.name)
            dataset = relative.parts[0] if relative.parts else "unknown"
            sample = "/".join(relative.parts[1:]) if len(relative.parts) > 1 else report.parent.name
            rows.append(make_validation_summary_row(
                dataset=dataset,
                sample=sample,
                status="ERROR",
                zarr_path=report.parent / "image.ome.zarr",
                detail=f"Could not parse validation report: {type(exc).__name__}: {exc}",
            ))
    return rows


def _deduplicate_summary_rows(rows: Iterable[ValidationSummaryRow]) -> list[ValidationSummaryRow]:
    """Remove duplicate summary rows while retaining the latest supplied row."""
    indexed: dict[tuple[str, str, str, str], ValidationSummaryRow] = {}
    for row in rows:
        key = (row.dataset, row.sample, row.source_path, row.zarr_path)
        indexed[key] = row
    return sorted(
        indexed.values(),
        key=lambda row: (row.dataset.lower(), row.sample.lower(), row.source_path.lower()),
    )


def write_validation_summary(
    rows: Iterable[ValidationSummaryRow],
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Write a compact validation summary without reproducing per-check details."""
    normalized = _deduplicate_summary_rows(rows)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = {status: sum(row.status == status for row in normalized) for status in ("PASS", "FAIL", "ERROR", "SKIPPED")}
    total_checks_passed = sum(row.checks_passed for row in normalized)
    total_checks_failed = sum(row.checks_failed for row in normalized)
    lines = [
        title,
        "=" * 120,
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Images represented: {len(normalized)}",
        f"PASS: {counts['PASS']}",
        f"FAIL: {counts['FAIL']}",
        f"ERROR: {counts['ERROR']}",
        f"SKIPPED: {counts['SKIPPED']}",
        f"Validation checks passed: {total_checks_passed}",
        f"Validation checks failed: {total_checks_failed}",
        "",
        f"{'STATUS':<8} | {'DATASET':<16} | {'SAMPLE':<40} | {'PASSED':>6} | {'FAILED':>6} | SOURCE CZI | OME-ZARR | DETAIL",
        "-" * 120,
    ]
    for row in normalized:
        lines.append(
            f"{row.status:<8} | {row.dataset:<16} | {row.sample:<40} | "
            f"{row.checks_passed:>6} | {row.checks_failed:>6} | "
            f"{row.source_path} | {row.zarr_path} | {row.detail}"
        )
    lines.extend([
        "",
        "Detailed per-check results remain in each sample's ome_zarr_validation.txt file.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_dataset_and_global_validation_summaries(
    results_img_root: str | Path,
    *,
    extra_rows: Iterable[ValidationSummaryRow] = (),
) -> tuple[dict[str, Path], Path]:
    """Refresh every dataset summary and the global summary below ``results/img``."""
    root = Path(results_img_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = _deduplicate_summary_rows([*scan_validation_reports(root), *list(extra_rows)])

    dataset_paths: dict[str, Path] = {}
    datasets = sorted({row.dataset for row in rows if row.dataset}, key=str.lower)
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.dataset == dataset]
        dataset_path = root / dataset / DATASET_SUMMARY_FILENAME
        dataset_paths[dataset] = write_validation_summary(
            dataset_rows,
            dataset_path,
            title=f"PFT OME-Zarr validation summary: {dataset}",
        )

    global_path = write_validation_summary(
        rows,
        root / GLOBAL_SUMMARY_FILENAME,
        title="PFT global OME-Zarr processing and validation summary",
    )
    return dataset_paths, global_path

def validate_or_raise(
    zarr_path: str | Path,
    source_arr: np.ndarray,
    source_meta: CziMeta,
    *,
    report_path: str | Path | None = None,
    print_terminal: bool = True,
) -> ValidationResult:
    """Run complete validation and raise ``OMEZarrValidationError`` when any check fails."""
    result = validate_ome_zarr(
        zarr_path,
        source_arr,
        source_meta,
        report_path=report_path,
        print_terminal=print_terminal,
    )
    if not result.passed:
        names = ", ".join(check.name for check in result.failed_checks[:8])
        if len(result.failed_checks) > 8:
            names += ", ..."
        raise OMEZarrValidationError(
            f"OME-Zarr validation failed for {result.zarr_path}. Failed checks: {names}. "
            f"See {result.report_path}"
        )
    return result
