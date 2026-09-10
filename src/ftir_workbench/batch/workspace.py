"""Verified independent workspaces: explicit JSON records and non-pickle NPY arrays."""
from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import fields
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig, PipelineConfig
from ftir_baseline.io import TextImportOptions
from ftir_baseline.models import BaselineResult, SpectrumSet, thaw_mapping
from ftir_baseline.normalization import NormalizationResult
from ftir_baseline.pipeline import PipelineResult
from ftir_baseline.qc import QCResult
from ftir_baseline.units import UnitConversionRecord, convert_to_absorbance

from .fingerprints import implementation_fingerprint, stage_fingerprint
from .importing import import_sources
from .models import (
    BatchError,
    BatchWorkspace,
    ImportedSource,
    PreparationConfig,
    SpectrumProcessingState,
    SpectrumRecord,
    StageSnapshot,
)
from .recipes import coarse_config, preparation_from_config, validate_preparation
from .state import _validate_snapshot, get_ready_snapshot

MAX_MEMBERS = 20000
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def _failure(message: str) -> BatchError:
    return BatchError("WORKSPACE_INTEGRITY_FAILED", message)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False).encode("utf-8")


def _read_json(payload: bytes) -> Any:
    def reject_constant(value: str) -> Any:
        raise _failure(f"non-standard JSON numeric constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise _failure("duplicate JSON object key")
        return result

    return json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_object)


def make_archive(members: dict[str, bytes], artifact_type: str) -> bytes:
    manifest = {
        "artifact_type": artifact_type, "schema_version": "1.0",
        "integrity_notice": "SHA-256 checks integrity; it is not an authentication signature.",
        "files": {name: {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(members.items())},
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted({**members, "manifest.json": json_bytes(manifest)}.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return stream.getvalue()


def read_verified_archive(payload: bytes, artifact_type: str) -> dict[str, bytes]:
    """Validate the whole member set before reading any archive-owned content."""
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise _failure("archive exceeds compressed size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
                raise _failure("archive exceeds member count or expanded size limit")
            names = [item.filename for item in infos]
            if len(set(names)) != len(names):
                raise _failure("duplicate archive member")
            for item in infos:
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                        or ":" in item.filename or str(path) != item.filename or item.is_dir()
                        or item.file_size > MAX_MEMBER_BYTES or item.flag_bits & 1
                        or stat.S_ISLNK(item.external_attr >> 16)
                        or item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}):
                    raise _failure("unsafe archive member or resource limit")
            members = {item.filename: archive.read(item) for item in infos}
        manifest = _read_json(members.pop("manifest.json"))
        if manifest["artifact_type"] != artifact_type or manifest["schema_version"] != "1.0":
            raise _failure("unsupported artifact type or schema")
        if set(manifest["files"]) != set(members):
            raise _failure("manifest member set mismatch")
        for name, data in members.items():
            if manifest["files"][name] != {
                "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
            }:
                raise _failure(f"member size or SHA-256 mismatch: {name}")
        return members
    except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile, RuntimeError, zlib.error, EOFError, RecursionError) as exc:
        if isinstance(exc, BatchError):
            raise
        raise _failure(f"invalid archive: {type(exc).__name__}") from exc


class _Arrays:
    def __init__(self, members: dict[str, bytes]) -> None:
        self.members = members
        self.counter = 0
        self.used: set[str] = set()

    def encode(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            name = f"arrays/{self.counter:06d}.npy"
            self.counter += 1
            stream = io.BytesIO()
            np.save(stream, np.asarray(value, dtype=np.float64), allow_pickle=False)
            self.members[name] = stream.getvalue()
            return {"__array__": name}
        if isinstance(value, Mapping):
            return {str(key): self.encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.encode(item) for item in value]
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return {"__nonfinite__": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise _failure(f"unsupported workspace value: {type(value).__name__}")

    def decode(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"__array__"}:
                name = value["__array__"]
                if not isinstance(name, str) or not name.startswith("arrays/") or not name.endswith(".npy"):
                    raise _failure("invalid array reference")
                self.used.add(name)
                stream = io.BytesIO(self.members[name])
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise _failure("unsupported NPY format")
                if (dtype != np.dtype("float64") or fortran or len(shape) > 3
                        or any(not isinstance(n, int) or n < 0 for n in shape)
                        or math.prod(shape) * 8 != len(stream.getbuffer()) - stream.tell()):
                    raise _failure("invalid NPY dtype, shape or byte count")
                stream.seek(0)
                return np.load(stream, allow_pickle=False)
            if set(value) == {"__nonfinite__"}:
                if value["__nonfinite__"] not in {"nan", "inf", "-inf"}:
                    raise _failure("invalid nonfinite metadata marker")
                return float(value["__nonfinite__"])
            return {key: self.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        return value


def _fields(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _spectrum_payload(value: SpectrumSet) -> dict[str, Any]:
    return _fields(value)


def _result_payload(result: PipelineResult) -> dict[str, Any]:
    return {
        "raw_input": _spectrum_payload(result.raw_input),
        "absorbance_full": _spectrum_payload(result.absorbance_full),
        "absorbance_selected": _spectrum_payload(result.absorbance_selected),
        "baseline_estimation_spectra": result.baseline_estimation_spectra,
        "baseline": _fields(result.baseline), "normalization": _fields(result.normalization),
        "unit_conversion": result.unit_conversion.to_dict(), "qc": _fields(result.qc),
        "config": result.config.to_dict(), "recipe": result.recipe,
        "input_sha256": result.input_sha256, "software_version": result.software_version,
        "warnings": result.warnings, "sensitivity_branches": result.sensitivity_branches,
    }


def _load_result(data: dict[str, Any]) -> PipelineResult:
    values = dict(data)
    for key in ("raw_input", "absorbance_full", "absorbance_selected"):
        values[key] = SpectrumSet(**values[key])
    values["baseline"] = BaselineResult(**values["baseline"])
    values["normalization"] = NormalizationResult(**values["normalization"])
    values["qc"] = QCResult(**values["qc"])
    conversion = dict(values["unit_conversion"])
    conversion["repaired_indices"] = tuple(tuple(row) for row in conversion["repaired_indices"])
    values["unit_conversion"] = UnitConversionRecord(**conversion)
    values["config"] = PipelineConfig(**values["config"])
    return PipelineResult(**values)


def _snapshot_payload(snapshot: StageSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {**_fields(snapshot), "config": snapshot.config.to_dict(),
            "result": _result_payload(snapshot.result)}


def _load_snapshot(data: dict[str, Any] | None) -> StageSnapshot | None:
    if data is None:
        return None
    return StageSnapshot(**{**data, "config": PipelineConfig(**data["config"]),
                            "result": _load_result(data["result"])})


def _editor_value(value: Any) -> Any:
    """Missing editor cells are JSON null, while their validation errors survive."""
    if isinstance(value, Mapping):
        return {key: _editor_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_editor_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_batch_workspace(workspace: BatchWorkspace) -> bytes:
    """Save drafts and confirmed results independently; previews are deliberately omitted."""
    members: dict[str, bytes] = {}
    arrays = _Arrays(members)
    sources = {}
    for index, (source_id, source) in enumerate(workspace.sources.items()):
        name = f"sources/{index:06d}.txt"
        members[name] = source.original_bytes
        sources[source_id] = {key: value for key, value in _fields(source).items()
                              if key != "original_bytes"}
        sources[source_id]["bytes_member"] = name
    states = {}
    for spectrum_id, state in workspace.states.items():
        data = _fields(state)
        for key in ("preparation_draft", "preparation_committed", "coarse_draft",
                    "coarse_committed", "fine_draft", "fine_committed"):
            data[key] = None if data[key] is None else data[key].to_dict()
        for key in ("coarse_snapshot", "fine_snapshot"):
            data[key] = _snapshot_payload(data[key])
        data["coarse_preview"] = data["fine_preview"] = None
        data["display_preferences"] = _editor_value(data["display_preferences"])
        states[spectrum_id] = data
    payload = {
        "artifact_type": "independent_baseline_workspace", "schema_version": "1.0",
        "workflow_mode": workspace.workflow_mode, "workspace_id": workspace.workspace_id,
        "saved_implementation_fingerprint": implementation_fingerprint(),
        "sources": sources, "records": {key: _fields(value) for key, value in workspace.records.items()},
        "states": states, "display_order": workspace.display_order,
        "selected_spectrum_id": workspace.selected_spectrum_id,
        "import_issues": workspace.import_issues, "export_history": workspace.export_history,
        "last_export_summary": workspace.last_export_summary,
    }
    members["workspace.json"] = json_bytes(arrays.encode(payload))
    return make_archive(members, "independent_baseline_workspace")


def _validate_sources(workspace: BatchWorkspace) -> None:
    for source_id, source in workspace.sources.items():
        if source.source_id != source_id:
            raise _failure("source ID mapping mismatch")
        parsed = BatchWorkspace()
        ids = import_sources(parsed, [(source.original_filename, source.original_bytes)],
                             input_unit=source.default_input_unit,
                             options=TextImportOptions(**dict(source.import_options)))
        if not ids:
            raise _failure("saved source no longer passes the public text parser")
        reparsed_source = next(iter(parsed.sources.values()))
        if thaw_mapping(source.import_probe) != thaw_mapping(reparsed_source.import_probe):
            raise _failure("saved parser probe differs from reparsed source")
        columns = {parsed.records[key].original_column_index: parsed.records[key] for key in ids}
        for record in workspace.records.values():
            if record.source_id != source_id:
                continue
            candidate = columns.get(record.original_column_index)
            if (candidate is None or candidate.original_column_label != record.original_column_label
                    or not np.array_equal(candidate.wavenumber, record.wavenumber)
                    or not np.array_equal(candidate.raw_intensity, record.raw_intensity)):
                raise _failure("source column mapping or original arrays mismatch")


def _validate_result(workspace: BatchWorkspace, spectrum_id: str, snapshot: StageSnapshot) -> None:
    _validate_snapshot(workspace, spectrum_id, snapshot)
    result, config = snapshot.result, snapshot.config
    raw, full, selected = result.raw_input, result.absorbance_full, result.absorbance_selected
    # These checks verify saved algebra and interpretation; no fitting or smoothing is executed.
    converted = convert_to_absorbance(raw.spectra, config.input_unit,
                                      transmittance_floor=config.transmittance_floor)
    x = validate_preparation(workspace.records[spectrum_id], preparation_from_config(config))
    mask = np.isin(full.wavenumber, x)
    recipe = result.recipe_dict()
    if (recipe.get("config") != config.to_dict()
            or recipe.get("input_sha256") != snapshot.input_sha256
            or recipe.get("input_source_name") != raw.source_name
            or recipe.get("input_metadata") != raw.mutable_metadata()
            or any(recipe.get(key) != value for key, value in config.to_dict().items())
            or recipe.get("unit_conversion_record") != result.unit_conversion.to_dict()
            or recipe.get("quality_control") != result.qc.as_dict()):
        raise _failure("saved pipeline recipe mismatch")
    checks = (np.array_equal(full.wavenumber, raw.wavenumber),
              np.array_equal(full.spectra, converted.absorbance),
              result.unit_conversion == converted.record,
              np.array_equal(selected.wavenumber, x),
              np.array_equal(selected.spectra, full.spectra[:, mask]),
              selected.intensity_unit == full.intensity_unit == "absorbance",
              result.baseline_estimation_spectra.shape == selected.spectra.shape,
              result.normalization.method == "none",
              result.normalization.optional_normalized is None,
              np.array_equal(result.view_data, result.analysis_data),
              np.array_equal(result.normalization.factors, np.ones(1)))
    if not all(checks):
        raise _failure("saved pipeline arrays, range or unit conversion mismatch")
    for spectrum in (raw, full, selected):
        if (spectrum.n_spectra != 1 or spectrum.perturbation_labels != ("independent",)
                or not np.array_equal(spectrum.perturbation, np.zeros(1))):
            raise _failure("saved adapter is not a singleton")
    if not config.baseline_smoothing.enabled and not np.array_equal(
        result.baseline_estimation_spectra, selected.spectra
    ):
        raise _failure("unsmoothed estimation channel differs from raw absorbance")
    for name in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
        if getattr(result.baseline, name).shape != selected.spectra.shape:
            raise _failure("baseline component shape differs from selected domain")
    expected_branches = {
        "uncorrected": selected.spectra,
        "coarse_only": selected.spectra - result.baseline.coarse_baseline,
        "coarse_plus_fine": result.baseline.corrected,
    }
    if set(result.sensitivity_branches) != set(expected_branches) or any(
        not np.array_equal(result.sensitivity_branches[key], value)
        for key, value in expected_branches.items()
    ):
        raise _failure("saved sensitivity branches disagree with components")
    if snapshot.stage == "coarse":
        if config != coarse_config(preparation_from_config(config), config.coarse_baseline):
            raise _failure("coarse recipe is not canonical fine-disabled")
        if snapshot.parent_coarse_fingerprint is not None or np.any(result.baseline.fine_baseline):
            raise _failure("coarse snapshot contains a fine parent or component")
    elif snapshot.stage == "fine":
        parent_config = coarse_config(preparation_from_config(config), config.coarse_baseline)
        parent = stage_fingerprint(workspace, spectrum_id, parent_config, "coarse",
                                   implementation=snapshot.parent_coarse_implementation_fingerprint)
        if parent != snapshot.parent_coarse_fingerprint:
            raise _failure("saved fine parent fingerprint mismatch")
    else:
        raise _failure("invalid stage")


def load_batch_workspace(bundle_bytes: bytes) -> BatchWorkspace:
    """Restore verified saved arrays without invoking the authoritative pipeline."""
    try:
        members = read_verified_archive(bundle_bytes, "independent_baseline_workspace")
        arrays = _Arrays(members)
        data = arrays.decode(_read_json(members["workspace.json"]))
        if (data["artifact_type"] != "independent_baseline_workspace"
                or data["schema_version"] != "1.0" or data["workflow_mode"] != "independent_batch"):
            raise _failure("unsupported workspace schema")
        if not isinstance(data["workspace_id"], str) or not data["workspace_id"].strip():
            raise _failure("workspace ID must be a nonempty string")
        workspace = BatchWorkspace(workspace_id=data["workspace_id"])
        used = {"workspace.json"} | arrays.used
        for key, source in data["sources"].items():
            values = dict(source)
            name = values.pop("bytes_member")
            if not isinstance(name, str) or not name.startswith("sources/"):
                raise _failure("invalid source member")
            used.add(name)
            workspace.sources[key] = ImportedSource(**values, original_bytes=members[name])
        workspace.records = {key: SpectrumRecord(**value) for key, value in data["records"].items()}
        for key, values in data["states"].items():
            values = dict(values)
            for name, model in (("preparation_draft", PreparationConfig),
                                ("preparation_committed", PreparationConfig),
                                ("coarse_draft", CoarseBaselineConfig),
                                ("coarse_committed", CoarseBaselineConfig),
                                ("fine_draft", FineBaselineConfig),
                                ("fine_committed", FineBaselineConfig)):
                values[name] = None if values[name] is None else model(**values[name])
            values["coarse_snapshot"] = _load_snapshot(values["coarse_snapshot"])
            values["fine_snapshot"] = _load_snapshot(values["fine_snapshot"])
            if values["coarse_preview"] is not None or values["fine_preview"] is not None:
                raise _failure("workspace must not silently restore a confirmable preview")
            workspace.states[key] = SpectrumProcessingState(**values)
        for key in ("display_order", "selected_spectrum_id", "import_issues", "export_history",
                    "last_export_summary"):
            setattr(workspace, key, data[key])
        if (used != set(members) or set(workspace.records) != set(workspace.states)
                or len(workspace.display_order) != len(set(workspace.display_order))
                or set(workspace.display_order) != set(workspace.records)
                or (workspace.selected_spectrum_id is not None
                    and workspace.selected_spectrum_id not in workspace.records)):
            raise _failure("workspace member, state or display-order mapping mismatch")
        _validate_sources(workspace)
        for key, record in workspace.records.items():
            state = workspace.states[key]
            if record.spectrum_id != key or record.source_id not in workspace.sources:
                raise _failure("record identity mismatch")
            if state.preparation_committed.input_unit != record.confirmed_input_unit:
                raise _failure("committed unit disagrees with record")
            validate_preparation(record, state.preparation_committed)
            for stage in ("coarse", "fine"):
                snapshot = getattr(state, f"{stage}_snapshot")
                committed = getattr(state, f"{stage}_committed")
                if (snapshot is None) != (committed is None):
                    raise _failure("confirmed recipe and snapshot presence mismatch")
                if snapshot is None:
                    continue
                if snapshot.stage != stage:
                    raise _failure("snapshot placed in the wrong stage")
                _validate_result(workspace, key, snapshot)
                if committed != getattr(snapshot.config, f"{stage}_baseline"):
                    raise _failure("confirmed recipe differs from saved snapshot")
                if not getattr(state, f"{stage}_stale"):
                    get_ready_snapshot(workspace, key, stage)
                if snapshot.implementation_fingerprint != implementation_fingerprint():
                    notice = "Restored from a saved older implementation; not recomputed or reconfirmed by the current pipeline."
                    if notice not in state.warnings:
                        state.warnings.append(notice)
            fine = state.fine_snapshot
            if state.fine_decision not in {"not_decided", "applied", "explicitly_skipped"}:
                raise _failure("invalid fine decision")
            if fine is None:
                if state.fine_decision != "not_decided" or state.fine_parent_coarse_fingerprint is not None:
                    raise _failure("fine decision has no matching saved result")
            elif (state.fine_decision == "not_decided"
                  or state.fine_parent_coarse_fingerprint != fine.parent_coarse_fingerprint):
                raise _failure("fine decision or parent mismatch")
            elif state.fine_decision == "explicitly_skipped" and (
                fine.config.fine_baseline.enabled or np.any(fine.result.baseline.fine_baseline)
            ):
                raise _failure("explicit skip contains active fine correction")
        return workspace
    except (KeyError, TypeError, ValueError, OSError, OverflowError, EOFError, RecursionError, zlib.error) as exc:
        if isinstance(exc, BatchError) and exc.code == "WORKSPACE_INTEGRITY_FAILED":
            raise
        raise _failure(f"workspace validation failed: {exc}") from exc
