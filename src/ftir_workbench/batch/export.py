"""Serialize individually confirmed stage arrays; exports never fit spectra."""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ftir_workbench.display_units import (
    convert_absorbance_for_display,
    derived_transmittance_filename,
)

from .models import BatchError, BatchWorkspace, Stage, StageSnapshot
from .state import draft_changes, get_ready_snapshot
from .workspace import json_bytes, make_archive, read_verified_archive

DerivedUnit = Literal["fraction_transmittance", "percent_transmittance"]


@dataclass(frozen=True, slots=True)
class BatchExportArtifact:
    zip_bytes: bytes
    filename: str
    metadata: dict[str, Any]
    report: list[dict[str, Any]]
    wide_csv_bytes: bytes | None
    wide_unavailable_reason: str | None


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")[:80] or "spectrum"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _csv_text(value: str) -> str:
    stripped = value.lstrip()
    if value.startswith(("\t", "\r", "\n")) or stripped.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([_csv_text(header) for header in headers])
    for row in rows:
        writer.writerow([format(float(value), ".17g") if isinstance(value, (float, np.floating))
                         else _csv_text(value) if isinstance(value, str) else value for value in row])
    return stream.getvalue().encode("utf-8")


def _select(workspace: BatchWorkspace, ids: Sequence[str], stage: Stage,
            ready_only: bool) -> tuple[list[tuple[str, StageSnapshot]], list[dict[str, Any]]]:
    if stage not in {"coarse", "fine"}:
        raise ValueError("stage must be coarse or fine")
    requested = list(dict.fromkeys(ids))
    if not requested:
        raise BatchError("EXPORT_NOT_READY", "select at least one spectrum")
    ready: list[tuple[str, StageSnapshot]] = []
    report: list[dict[str, Any]] = []
    for spectrum_id in requested:
        row: dict[str, Any] = {"spectrum_id": spectrum_id, "stage": stage}
        try:
            record = workspace.records[spectrum_id]
            state = workspace.states[spectrum_id]
            row.update(display_name=record.display_name, source_id=record.source_id,
                       source_column_index=record.original_column_index)
            changes = draft_changes(workspace, spectrum_id)
            relevant = ["preparation", "coarse"] + (["fine"] if stage == "fine" else [])
            invalid = {key: value for key, value in state.display_preferences.items()
                       if key.startswith("error_") and value}
            row["uncommitted_draft_not_used"] = any(changes[key] for key in relevant) or bool(invalid)
            row["invalid_editor_not_used"] = json_bytes(_plain(invalid)).decode("utf-8") if invalid else ""
            row["recorded_errors"] = " | ".join(state.errors)
            if record.excluded:
                raise BatchError("EXCLUDED", "spectrum explicitly excluded")
            snapshot = get_ready_snapshot(workspace, spectrum_id, stage)
            row.update(status="exported", reason="", stage_fingerprint=snapshot.fingerprint,
                       fine_decision=state.fine_decision if stage == "fine" else "not_applicable")
            ready.append((spectrum_id, snapshot))
        except (BatchError, KeyError) as exc:
            row.update(status="excluded", reason=str(exc),
                       reason_code=exc.code if isinstance(exc, BatchError) else "UNKNOWN_SPECTRUM")
        report.append(row)
    excluded = [row for row in report if row["status"] != "exported"]
    if excluded and not ready_only:
        details = "; ".join(f"{row['spectrum_id']}: {row['reason']}" for row in excluded)
        raise BatchError("EXPORT_NOT_READY", details)
    if not ready:
        raise BatchError("EXPORT_NOT_READY", "no confirmed, current results are ready")
    return ready, report


def can_export_wide(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, stage: Stage,
                    ready_only: bool = False) -> tuple[bool, str]:
    try:
        ready, _ = _select(workspace, spectrum_ids, stage, ready_only)
    except BatchError as exc:
        return False, str(exc)
    first = ready[0][1].result.absorbance_selected.wavenumber
    if any(not np.array_equal(first, snapshot.result.absorbance_selected.wavenumber)
           for _, snapshot in ready[1:]):
        return False, "Actual output axes differ in length, direction, range or coordinates; no alignment or interpolation is performed."
    return True, ""


def build_batch_export(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, stage: Stage,
                       ready_only: bool = False, include_wide: bool = False,
                       derived_units: Sequence[DerivedUnit] = ()) -> BatchExportArtifact:
    ready, report = _select(workspace, spectrum_ids, stage, ready_only)
    derived_units = tuple(dict.fromkeys(derived_units))
    if any(unit not in {"fraction_transmittance", "percent_transmittance"} for unit in derived_units):
        raise ValueError("derived units must be T or %T")
    first_axis = ready[0][1].result.absorbance_selected.wavenumber
    same_axis = all(np.array_equal(first_axis, item.result.absorbance_selected.wavenumber)
                    for _, item in ready)
    wide_reason = None if same_axis else (
        "Actual output axes differ in length, direction, range or coordinates; "
        "no alignment or interpolation is performed."
    )
    if include_wide and not same_axis:
        raise BatchError("AXIS_MISMATCH_FOR_WIDE_EXPORT", str(wide_reason))
    members: dict[str, bytes] = {}
    metadata: dict[str, Any] = {
        "artifact_type": "independent_baseline_batch", "schema_version": "1.0",
        "workflow_mode": "independent_batch", "stage": stage, "is_2d_ready": False,
        "workspace_id": workspace.workspace_id,
        "requested_spectrum_ids": list(dict.fromkeys(spectrum_ids)),
        "exported_spectrum_ids": [key for key, _ in ready],
        "excluded_items": [row for row in report if row["status"] != "exported"],
        "selection_policy": "ready_only" if ready_only else "strict",
        "import_issues": workspace.import_issues,
        "spectra": [], "wide_available": same_axis, "wide_unavailable_reason": wide_reason,
        "qc_applicability": {
            "cross_sample_time_continuity": "not_applicable",
            "internal_adapter_placeholder": "not an experimental perturbation",
        },
    }
    qc_rows = []
    wide_labels = []
    for spectrum_id, snapshot in ready:
        record, state = workspace.records[spectrum_id], workspace.states[spectrum_id]
        source, result = workspace.sources[record.source_id], snapshot.result
        name = f"{_safe_name(record.display_name)}__{_safe_name(spectrum_id)}"
        wide_labels.append(name)
        x = result.absorbance_selected.wavenumber
        y = result.analysis_data[0]
        spectra_name = f"spectra/{name}__{stage}_corrected_absorbance.csv"
        components_name = f"components/{name}__baseline_components.csv"
        members[spectra_name] = _table(["wavenumber_cm-1", "corrected_absorbance"],
                                       list(zip(x, y, strict=True)))
        members[components_name] = _table(
            ["wavenumber_cm-1", "raw_absorbance", "coarse_baseline", "fine_baseline",
             "total_baseline", "corrected_absorbance"],
            list(zip(x, result.absorbance_selected.spectra[0], result.baseline.coarse_baseline[0],
                     result.baseline.fine_baseline[0], result.baseline.total_baseline[0], y, strict=True)),
        )
        entry: dict[str, Any] = {
            "spectrum_id": spectrum_id, "display_name": record.display_name,
            "source_id": record.source_id, "original_filename": source.original_filename,
            "source_sha256": source.original_bytes_sha256,
            "source_column_index": record.original_column_index,
            "source_column_label": record.original_column_label,
            "scientific_input_sha256": snapshot.input_sha256,
            "stage_fingerprint": snapshot.fingerprint,
            "parent_coarse_fingerprint": snapshot.parent_coarse_fingerprint,
            "parent_coarse_implementation_fingerprint": snapshot.parent_coarse_implementation_fingerprint,
            "implementation_fingerprint": snapshot.implementation_fingerprint,
            "input_unit": snapshot.config.input_unit, "output_unit": "absorbance",
            "original_axis_direction": record.original_axis_direction,
            "output_axis_direction": result.absorbance_selected.axis_direction,
            "processing_range": list(snapshot.config.wavenumber_range),
            "fine_decision": state.fine_decision if stage == "fine" else "not_applicable",
            "spectra_csv": spectra_name, "components_csv": components_name,
            "unit_conversion": result.unit_conversion.to_dict(),
            "derived_outputs": [],
        }
        for unit in derived_units:
            converted = convert_absorbance_for_display(y, unit)
            filename = f"spectra/{name}__{derived_transmittance_filename(unit)}"
            members[filename] = (
                b"# Mathematically derived from corrected absorbance; not original instrument transmittance.\n"
                + _table(["wavenumber_cm-1", unit], list(zip(x, converted.values, strict=True)))
            )
            entry["derived_outputs"].append({"filename": filename, "formula": converted.formula,
                                              "warnings": list(converted.warnings)})
        metadata["spectra"].append(entry)
        core_recipe = result.recipe_dict()
        core_qc = core_recipe.pop("quality_control", {})
        recipe = {"spectrum": entry, "authoritative_pipeline_recipe": core_recipe,
                  "internal_adapter_qc": core_qc,
                  "qc_applicability": metadata["qc_applicability"]}
        members[f"recipes/{_safe_name(spectrum_id)}.json"] = json_bytes(_plain(recipe))
        metrics = result.qc.as_dict()["per_spectrum"]
        for metric, values in metrics.items():
            if metric in {"perturbation", "spectrum_index", "adjacent_baseline_rms"}:
                continue
            value = values[0]
            qc_rows.append([spectrum_id, metric, value if value is not None else "N/A",
                            "applicable" if value is not None else "not_applicable"])
        qc_rows.append([spectrum_id, "cross_sample_time_continuity", "N/A", "not_applicable"])
    wide = None
    if include_wide:
        wide = _table(["wavenumber_cm-1", *wide_labels],
                      list(zip(first_axis, *(item.result.analysis_data[0] for _, item in ready), strict=True)))
        members[f"{stage}_corrected_absorbance_wide.csv"] = wide
    report_headers = sorted({key for row in report for key in row})
    members["processing_report.csv"] = _table(report_headers, [
        [row.get(key, "") for key in report_headers] for row in report
    ])
    members["qc_summary.csv"] = _table(["spectrum_id", "metric", "value", "applicability"], qc_rows)
    members["batch_metadata.json"] = json_bytes(_plain(metadata))
    return BatchExportArtifact(
        zip_bytes=make_archive(members, "independent_baseline_batch"),
        filename=f"independent_baseline_{stage}_{_safe_name(workspace.workspace_id)[:12]}.zip",
        metadata=metadata, report=report, wide_csv_bytes=wide, wide_unavailable_reason=wide_reason,
    )


def verify_batch_export(payload: bytes) -> bool:
    try:
        read_verified_archive(payload, "independent_baseline_batch")
        return True
    except BatchError:
        return False
