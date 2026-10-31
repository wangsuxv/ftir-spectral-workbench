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

from .export_naming import (
    NAMING_POLICY_VERSION,
    Naming,
    allocate_bases,
    ensure_unique_names,
    get_naming_options,
    make_name_row,
    name_map_table,
    naming_plan_digest,
    validate_name_row,
)
from .models import BatchError, BatchWorkspace, Stage, StageSnapshot
from .state import draft_changes, get_ready_snapshot
from .workspace import _read_json, json_bytes, make_archive, read_verified_archive

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
            ready_only: bool, *, allow_empty: bool = False) -> tuple[list[tuple[str, StageSnapshot]], list[dict[str, Any]]]:
    if stage not in {"coarse", "fine"}:
        raise ValueError("stage must be coarse or fine")
    requested = list(dict.fromkeys(ids))
    if not requested and not allow_empty:
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
    if not ready and not allow_empty:
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


def _name_plan(workspace: BatchWorkspace, ready: Sequence[tuple[str, StageSnapshot]],
               naming: Naming, derived_units: Sequence[DerivedUnit]) -> tuple[list[dict[str, Any]], str]:
    bases = allocate_bases(workspace, naming)
    legacy = get_naming_options(workspace, naming)["policy"] == "legacy_id"
    rows = []
    for sid, snapshot in ready:
        row = make_name_row(
            bases[sid], branch="coarse" if snapshot.stage == "coarse" else "baseline",
            quantity="absorbance", purpose="scientific", fingerprint=snapshot.fingerprint,
            parent_fingerprint=snapshot.parent_coarse_fingerprint,
            recipe_json=f"recipes/{_safe_name(sid)}.json", included_as="requested",
            components=True,
            derived_filenames=[derived_transmittance_filename(unit) for unit in derived_units],
        )
        if legacy:
            name = f"{_safe_name(workspace.records[sid].display_name)}__{_safe_name(sid)}"
            row.update(spectra_csv=f"spectra/{name}__{snapshot.stage}_corrected_absorbance.csv",
                       components_csv=f"components/{name}__baseline_components.csv",
                       wide_column_label=name,
                       derived_outputs={derived_transmittance_filename(unit): f"spectra/{name}__{derived_transmittance_filename(unit)}" for unit in derived_units})
        rows.append(row)
    if not legacy:
        ensure_unique_names(rows)
    return rows, naming_plan_digest(bases, rows)


def preview_batch_export(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, stage: Stage,
                         naming: Naming = None, ready_only: bool = False,
                         derived_units: Sequence[DerivedUnit] = ()) -> dict[str, Any]:
    """Describe confirmed paths and exclusions without serializing or calculating."""
    units = tuple(dict.fromkeys(derived_units))
    if any(unit not in {"fraction_transmittance", "percent_transmittance"} for unit in units):
        raise ValueError("derived units must be T or %T")
    ready, report = _select(workspace, spectrum_ids, stage, True, allow_empty=True)
    rows, digest = _name_plan(workspace, ready, naming, units)
    return {"name_map": rows, "report": report, "naming_plan_digest": digest,
            "exportable": bool(ready) and (ready_only or all(row["status"] == "exported" for row in report))}


def build_batch_export(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, stage: Stage,
                       ready_only: bool = False, include_wide: bool = False,
                       derived_units: Sequence[DerivedUnit] = (),
                       naming: Naming = None) -> BatchExportArtifact:
    ready, report = _select(workspace, spectrum_ids, stage, ready_only)
    derived_units = tuple(dict.fromkeys(derived_units))
    if any(unit not in {"fraction_transmittance", "percent_transmittance"} for unit in derived_units):
        raise ValueError("derived units must be T or %T")
    name_rows, _ = _name_plan(workspace, ready, naming, derived_units)
    legacy = get_naming_options(workspace, naming)["policy"] == "legacy_id"
    names = {row["spectrum_id"]: row for row in name_rows}
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
    if not legacy:
        metadata.update(schema_version="1.1", naming_policy_version=NAMING_POLICY_VERSION,
                        naming_policy=get_naming_options(workspace, naming)["policy"],
                        name_map=name_rows, name_map_csv="export_name_map.csv",
                        processing_report=report, wide_csv=None, wide_headers=[])
    qc_rows = []
    wide_labels = []
    for spectrum_id, snapshot in ready:
        record, state = workspace.records[spectrum_id], workspace.states[spectrum_id]
        source, result = workspace.sources[record.source_id], snapshot.result
        plan = names[spectrum_id]
        wide_labels.append(plan["wide_column_label"])
        x = result.absorbance_selected.wavenumber
        y = result.analysis_data[0]
        spectra_name = plan["spectra_csv"]
        components_name = plan["components_csv"]
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
            filename = plan["derived_outputs"][derived_transmittance_filename(unit)]
            members[filename] = (
                b"# Mathematically derived from corrected absorbance; not original instrument transmittance.\n"
                + _table(["wavenumber_cm-1", unit], list(zip(x, converted.values, strict=True)))
            )
            entry["derived_outputs"].append({"filename": filename, "formula": converted.formula,
                                              "warnings": list(converted.warnings)})
        if not legacy:
            entry["recipe_json"] = plan["recipe_json"]
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
        if not legacy:
            metadata.update(wide_csv=f"{stage}_corrected_absorbance_wide.csv", wide_headers=wide_labels)
    report_headers = sorted({key for row in report for key in row})
    members["processing_report.csv"] = _table(report_headers, [
        [row.get(key, "") for key in report_headers] for row in report
    ])
    members["qc_summary.csv"] = _table(["spectrum_id", "metric", "value", "applicability"], qc_rows)
    if not legacy:
        members["export_name_map.csv"] = _table(*name_map_table(name_rows))
    members["batch_metadata.json"] = json_bytes(_plain(metadata))
    return BatchExportArtifact(
        zip_bytes=make_archive(members, "independent_baseline_batch"),
        filename=f"independent_baseline_{stage}_{_safe_name(workspace.workspace_id)[:12]}.zip",
        metadata=metadata, report=report, wide_csv_bytes=wide, wide_unavailable_reason=wide_reason,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BatchError("INVALID_BATCH_EXPORT", message)


def _read_table(payload: bytes, headers: Sequence[str]) -> np.ndarray[Any, Any]:
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
    _require(bool(rows) and rows[0] == [_csv_text(value) for value in headers], "CSV header mismatch")
    values = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == len(headers) and values.shape[0] >= 2
             and bool(np.all(np.isfinite(values))), "invalid numeric CSV")
    return values


def verify_batch_export(payload: bytes) -> bool:
    """Verify old archive integrity or new names/paths/identities and array agreement.

    Neither format refits the baseline; hashes are integrity checks, not signatures.
    """
    try:
        members = read_verified_archive(payload, "independent_baseline_batch")
        metadata = _read_json(members["batch_metadata.json"])
        schema = metadata["schema_version"]
        _require(schema in {"1.0", "1.1"}, "unsupported batch export schema")
        if schema == "1.0":
            _require("export_name_map.csv" not in members
                     and not {"name_map", "name_map_csv", "naming_policy", "naming_policy_version"}.intersection(metadata),
                     "naming fields require schema 1.1")
            return True
        _require(metadata["artifact_type"] == "independent_baseline_batch"
                 and metadata["workflow_mode"] == "independent_batch"
                 and metadata["is_2d_ready"] is False, "invalid artifact/mode")
        _require(metadata["stage"] in {"coarse", "fine"}, "invalid stage")
        _require(metadata["naming_policy_version"] == NAMING_POLICY_VERSION
                 and metadata["naming_policy"] in {"original", "display"}
                 and metadata["name_map_csv"] == "export_name_map.csv", "invalid naming contract")
        rows, entries = metadata["name_map"], metadata["spectra"]
        _require(isinstance(rows, list) and isinstance(entries, list) and len(rows) == len(entries) > 0,
                 "invalid name map size")
        ensure_unique_names(rows)
        _require(members["export_name_map.csv"] == _table(*name_map_table(rows)), "CSV/JSON name map mismatch")
        _require([entry["spectrum_id"] for entry in entries] == metadata["exported_spectrum_ids"]
                 == [row["spectrum_id"] for row in rows], "export identity order mismatch")
        _require(len(set(metadata["exported_spectrum_ids"])) == len(rows)
                 and set(metadata["exported_spectrum_ids"]) <= set(metadata["requested_spectrum_ids"]),
                 "duplicate or unrequested identities")
        used = {"batch_metadata.json", "export_name_map.csv", "qc_summary.csv", "processing_report.csv"}
        arrays = []
        for row, entry in zip(rows, entries, strict=True):
            _require(row["policy"] == metadata["naming_policy"], "inconsistent name policy")
            derived = row["derived_outputs"]
            allowed_derived = {derived_transmittance_filename(unit) for unit in ("fraction_transmittance", "percent_transmittance")}
            _require(isinstance(derived, dict) and derived.keys() <= allowed_derived,
                     "invalid derived output type")
            validate_name_row(
                row, entry, branch="coarse" if metadata["stage"] == "coarse" else "baseline",
                quantity="absorbance", purpose="scientific", fingerprint=entry["stage_fingerprint"],
                parent_fingerprint=entry["parent_coarse_fingerprint"],
                recipe_json=f"recipes/{_safe_name(entry['spectrum_id'])}.json", included_as="requested",
                components=True, derived_filenames=list(derived),
            )
            for key in ("spectra_csv", "components_csv", "recipe_json"):
                _require(entry[key] == row[key] and row[key] not in used, "path/identity mismatch")
                used.add(row[key])
            recipe = _read_json(members[row["recipe_json"]])
            _require(recipe["spectrum"] == entry, "recipe/source mismatch")
            values = _read_table(members[row["spectra_csv"]], ["wavenumber_cm-1", "corrected_absorbance"])
            parts = _read_table(members[row["components_csv"]], ["wavenumber_cm-1", "raw_absorbance", "coarse_baseline", "fine_baseline", "total_baseline", "corrected_absorbance"])
            _require(np.array_equal(values[:, 0], parts[:, 0]) and np.array_equal(values[:, 1], parts[:, -1]),
                     "spectrum/components disagreement")
            x = values[:, 0]
            _require(bool(np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0))
                     and entry["output_axis_direction"] == ("ascending" if x[0] < x[-1] else "descending"),
                     "axis direction mismatch")
            arrays.append(values)
            _require([item["filename"] for item in entry["derived_outputs"]] == list(derived.values()),
                     "derived path mapping mismatch")
            for filename, path in derived.items():
                unit = "percent_transmittance" if "percent" in filename else "fraction_transmittance"
                derived_values = _read_table(members[path].split(b"\n", 1)[1], ["wavenumber_cm-1", unit])
                _require(np.array_equal(derived_values[:, 0], x) and path not in used, "derived axis/path mismatch")
                used.add(path)
        same = all(np.array_equal(arrays[0][:, 0], item[:, 0]) for item in arrays)
        _require(metadata["wide_available"] is same, "wide availability mismatch")
        wide_path = metadata["wide_csv"]
        if wide_path is not None:
            labels = [row["wide_column_label"] for row in rows]
            _require(same and wide_path == f"{metadata['stage']}_corrected_absorbance_wide.csv"
                     and metadata["wide_headers"] == labels, "wide path/header mismatch")
            values = _read_table(members[wide_path], ["wavenumber_cm-1", *labels])
            expected = np.column_stack([arrays[0][:, 0], *(item[:, 1] for item in arrays)])
            _require(np.array_equal(values, expected), "wide values differ from per-spectrum arrays")
            used.add(wide_path)
        else:
            _require(metadata["wide_headers"] == [], "unexpected wide headers")
        report = metadata["processing_report"]
        _require([item["spectrum_id"] for item in report] == metadata["requested_spectrum_ids"]
                 and len(set(metadata["requested_spectrum_ids"])) == len(report)
                 and all(item["stage"] == metadata["stage"] and item["status"] in {"exported", "excluded"} for item in report),
                 "processing report request/stage mismatch")
        exported = [item for item in report if item["status"] == "exported"]
        _require([item["spectrum_id"] for item in exported] == metadata["exported_spectrum_ids"]
                 and metadata["excluded_items"] == [item for item in report if item["status"] != "exported"],
                 "processing report selection mismatch")
        _require(metadata["selection_policy"] in {"strict", "ready_only"}
                 and (metadata["selection_policy"] != "strict" or not metadata["excluded_items"]),
                 "selection policy mismatch")
        for item, entry in zip(exported, entries, strict=True):
            _require(all(item[key] == entry[key] for key in ("spectrum_id", "display_name", "source_id", "source_column_index", "stage_fingerprint", "fine_decision")),
                     "processing report spectrum identity mismatch")
        headers = sorted({key for item in report for key in item})
        _require(members["processing_report.csv"] == _table(headers, [[item.get(key, "") for key in headers] for item in report]),
                 "processing report mismatch")
        _require(used == set(members), "unexpected or missing export members")
        return True
    except (BatchError, ValueError, TypeError, KeyError, AttributeError, IndexError, UnicodeError, csv.Error):
        return False
