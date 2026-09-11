"""Confirmed ordinary postprocessing bundles and an explicit replay verifier.

Export is snapshot serialization only. Verification can separately replay S/N
from exported parents; a baseline anchor is checked for integrity, never refit.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import FloatArray
from ftir_workbench.post_baseline_smoothing import smooth_spectral_arrays, validate_spectral_arrays

from .export import BatchExportArtifact, _safe_name, _table
from .fingerprints import json_fingerprint, source_selection_fingerprint
from .models import BatchError, BatchWorkspace, StageSnapshot
from .normalization_adapter import normalize_spectral_arrays
from .postprocessing import (
    BRANCHES,
    PARENTS,
    PostprocessSnapshot,
    _request_payload,
    array_hash,
    branch_arrays,
    branch_status,
    dependency_versions,
    effective_config,
    get_branch,
    plain,
)
from .recipes import coarse_config, preparation_from_config, require_independent_config
from .state import draft_changes
from .workspace import _read_json, json_bytes, make_archive, read_verified_archive

ARTIFACT_TYPE = "ordinary_spectrum_postprocess_bundle"
_AXIS_REASON = (
    "Actual output axes differ in length, direction, range or coordinates; "
    "no alignment or interpolation is performed."
)
_BASELINE_NOTICE = (
    "Baseline anchor integrity and recorded lineage only; the baseline is not refitted. "
    "Raw source bytes are not included. Hashes are not signatures or proof of source authenticity."
)
_BASELINE_VERSION_SCOPE = (
    "The baseline snapshot records its ftir_baseline software version and implementation digest. "
    "Legacy baseline snapshots did not separately store numerical dependency versions; "
    "exporter_versions describes export time, not baseline calculation time."
)
Snapshot = StageSnapshot | PostprocessSnapshot


def _source(workspace: BatchWorkspace, sid: str) -> dict[str, Any]:
    record = workspace.records[sid]
    source = workspace.sources[record.source_id]
    return {
        "spectrum_id": sid,
        "source_id": record.source_id,
        "display_name": record.display_name,
        "original_filename": source.original_filename,
        "source_sha256": source.original_bytes_sha256,
        "source_column_index": record.original_column_index,
        "source_column_label": record.original_column_label,
        "original_axis_direction": record.original_axis_direction,
        "original_point_count": int(record.wavenumber.size),
        "import_options": plain(source.import_options),
        "source_selection_sha256": source_selection_fingerprint(workspace, sid),
    }


def _select(
    workspace: BatchWorkspace,
    spectrum_ids: Sequence[str],
    branches: Sequence[str],
    valid_only: bool,
) -> tuple[list[tuple[str, str, Snapshot]], list[dict[str, Any]]]:
    if isinstance(spectrum_ids, str) or isinstance(branches, str):
        raise BatchError("EXPORT_NOT_READY", "spectrum_ids and branches must be explicit sequences")
    ids, selected_branches = list(dict.fromkeys(spectrum_ids)), list(dict.fromkeys(branches))
    if (
        not ids
        or not selected_branches
        or any(branch not in BRANCHES for branch in selected_branches)
    ):
        raise BatchError("EXPORT_NOT_READY", "select at least one spectrum and supported branch")
    ready: list[tuple[str, str, Snapshot]] = []
    report: list[dict[str, Any]] = []
    for sid in ids:
        for branch in selected_branches:
            row: dict[str, Any] = {
                "spectrum_id": sid,
                "branch": branch,
                "status": "missing",
                "reason": "",
                "reason_code": "",
                "draft_modified": False,
                "recorded_errors": [],
                "invalid_editor_not_used": {},
            }
            try:
                row.update(_source(workspace, sid))
                state = workspace.states[sid]
                pending = draft_changes(workspace, sid)
                row["draft_modified"] = any(pending.values())
                row["recorded_errors"] = list(state.errors)
                row["invalid_editor_not_used"] = {
                    key: value
                    for key, value in state.display_preferences.items()
                    if key.startswith("error_") and value
                }
                if workspace.records[sid].excluded:
                    raise BatchError("EXCLUDED", "spectrum explicitly excluded")
                status = branch_status(workspace, sid, branch)
                row["draft_modified"] |= status["draft_modified"]
                post = workspace.postprocessing.get(sid)
                if branch != "baseline" and post is not None:
                    branch_state = getattr(post, branch)
                    row["recorded_errors"].extend(branch_state.errors)
                    row["invalid_editor_not_used"].update(
                        {
                            key: value
                            for key, value in post.display_preferences.items()
                            if key.startswith("error_") and value
                        }
                    )
                row["draft_modified"] |= bool(row["invalid_editor_not_used"])
                snapshot = get_branch(workspace, sid, branch)
                row.update(status="exported", fingerprint=snapshot.fingerprint)
                ready.append((sid, branch, snapshot))
            except (ValueError, TypeError, KeyError) as exc:
                code = exc.code if isinstance(exc, BatchError) else "UNKNOWN_SPECTRUM"
                status_name = (
                    "excluded"
                    if code == "EXCLUDED"
                    else (
                        "stale"
                        if "STALE" in code
                        else "failed"
                        if row["recorded_errors"]
                        else "missing"
                    )
                )
                row.update(status=status_name, reason=str(exc), reason_code=code)
            report.append(plain(row))
    failures = [row for row in report if row["status"] != "exported"]
    if failures and not valid_only:
        raise BatchError(
            "EXPORT_NOT_READY",
            "; ".join(f"{row['spectrum_id']}/{row['branch']}: {row['reason']}" for row in failures),
        )
    if not ready:
        raise BatchError(
            "EXPORT_NOT_READY", "no requested confirmed, current branches are available"
        )
    return ready, report


def can_export_postprocessing_wide(
    workspace: BatchWorkspace,
    spectrum_ids: Sequence[str],
    *,
    branches: Sequence[str] = ("baseline",),
    valid_only: bool = False,
) -> tuple[bool, str]:
    try:
        ready, _ = _select(workspace, spectrum_ids, branches, valid_only)
    except BatchError as exc:
        return False, str(exc)
    first = branch_arrays(ready[0][2])[0]
    same = all(np.array_equal(first, branch_arrays(snapshot)[0]) for _, _, snapshot in ready)
    return (True, "") if same else (False, _AXIS_REASON)


def _stage_request(workspace: BatchWorkspace, sid: str, snapshot: StageSnapshot) -> dict[str, Any]:
    source = _source(workspace, sid)
    return {
        "workspace_id": workspace.workspace_id,
        "spectrum_id": sid,
        "source_id": source["source_id"],
        "original_filename": source["original_filename"],
        "original_column_label": source["source_column_label"],
        "source_selection_sha256": source["source_selection_sha256"],
        "scientific_input_sha256": snapshot.input_sha256,
        "workflow_mode": "independent_batch",
        "stage": "fine",
        "config": snapshot.config.to_dict(),
        "parent_coarse_fingerprint": snapshot.parent_coarse_fingerprint,
        "implementation": snapshot.implementation_fingerprint,
    }


def _node_paths(
    source: Mapping[str, Any], branch: str, quantity: str
) -> tuple[str, str, str | None]:
    sid = str(source["spectrum_id"])
    stem = f"{_safe_name(str(source['display_name']))}__{_safe_name(sid)}__{hashlib.sha256(sid.encode()).hexdigest()[:12]}"
    suffix = branch + (
        "__minmax_scaled_intensity_display_only" if quantity == "minmax_scaled_intensity" else ""
    )
    return (
        f"spectra/{stem}__{suffix}.csv",
        f"recipes/{_safe_name(sid)}__{hashlib.sha256(sid.encode()).hexdigest()[:12]}__{branch}.json",
        f"components/{stem}__smoothed_removed_component.csv" if branch == "smoothed" else None,
    )


def _quantity_header(branch: str, quantity: str) -> str:
    if quantity == "minmax_scaled_intensity":
        return "minmax_scaled_intensity_display_only"
    if quantity == "normalized_intensity":
        return "normalized_intensity"
    return "smoothed_absorbance" if branch == "smoothed" else "corrected_absorbance"


def _serialize_node(
    workspace: BatchWorkspace,
    sid: str,
    branch: str,
    snapshot: Snapshot,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    source = _source(workspace, sid)
    x, y = branch_arrays(snapshot)
    quantity = "absorbance" if isinstance(snapshot, StageSnapshot) else snapshot.quantity
    spectra_path, recipe_path, removed_path = _node_paths(source, branch, quantity)
    node: dict[str, Any] = {
        "schema_version": "1.0",
        "workspace_id": workspace.workspace_id,
        "spectrum_id": sid,
        "source_id": source["source_id"],
        "source": source,
        "branch": branch,
        "quantity": quantity,
        "purpose": "scientific" if isinstance(snapshot, StageSnapshot) else snapshot.purpose,
        "fingerprint": snapshot.fingerprint,
        "spectra_csv": spectra_path,
        "recipe_json": recipe_path,
        "column_label": _quantity_header(branch, quantity),
        "x_sha256": array_hash(x),
        "y_sha256": array_hash(y),
        "point_count": int(x.size),
        "is_2d_ready": False,
        "output_axis_direction": "ascending" if x[0] < x[-1] else "descending",
        "parent_branch": None if branch == "baseline" else PARENTS[branch],
        "removed_csv": removed_path,
        "removed_sha256": None,
        "scale": None,
        "offset": None,
        "reference_details": {},
        "qc": {},
    }
    members = {
        spectra_path: _table(
            ["wavenumber_cm-1", node["column_label"]], list(zip(x, y[0], strict=True))
        )
    }
    if isinstance(snapshot, StageSnapshot):
        node.update(
            input_sha256=snapshot.input_sha256,
            baseline_fingerprint=snapshot.fingerprint,
            parent_fingerprint=None,
            request_fingerprint=snapshot.fingerprint,
            recipe=snapshot.config.to_dict(),
            effective_recipe=snapshot.config.to_dict(),
            implementation=snapshot.implementation_fingerprint,
            stage_request=_stage_request(workspace, sid, snapshot),
            parent_coarse_implementation=snapshot.parent_coarse_implementation_fingerprint,
            unit_conversion=plain(snapshot.result.unit_conversion.to_dict()),
            fine_decision=workspace.states[sid].fine_decision,
            warnings=list(snapshot.result.warnings),
            versions={"ftir_baseline": snapshot.result.software_version},
            version_scope=_BASELINE_VERSION_SCOPE,
            verification_scope=_BASELINE_NOTICE,
        )
    else:
        node.update(
            input_sha256=snapshot.input_sha256,
            baseline_fingerprint=snapshot.baseline.fingerprint,
            parent_fingerprint=snapshot.parent_fingerprint,
            request_fingerprint=snapshot.request_fingerprint,
            request_payload=_request_payload(snapshot),
            recipe=plain(snapshot.recipe),
            effective_recipe=plain(snapshot.effective_recipe),
            implementation=snapshot.implementation,
            versions=plain(snapshot.versions),
            scale=plain(snapshot.scale),
            offset=plain(snapshot.offset),
            reference_details=plain(snapshot.reference_details),
            qc=plain(snapshot.qc),
            warnings=list(snapshot.warnings),
        )
        if snapshot.removed_component is not None:
            assert removed_path is not None
            members[removed_path] = _table(
                ["wavenumber_cm-1", "smoothing_removed_component"],
                list(zip(x, snapshot.removed_component[0], strict=True)),
            )
            node["removed_sha256"] = array_hash(snapshot.removed_component)
    node["node_integrity_sha256"] = json_fingerprint(node)
    members[recipe_path] = json_bytes(node)
    return node, members


def _heterogeneous_warnings(nodes: Sequence[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for branch in BRANCHES:
        matches = [node for node in nodes if node["branch"] == branch]
        recipes = {json_fingerprint(node["effective_recipe"]) for node in matches}
        axes = {node["x_sha256"] for node in matches}
        if len(recipes) > 1 or len(axes) > 1:
            warnings.append(
                f"Heterogeneous {branch} recipes or output grids/ranges: spectra are preserved "
                "independently; this export does not establish quantitative comparability."
            )
    quantities = {node["quantity"] for node in nodes}
    if len(quantities) > 1:
        warnings.append(
            "The bundle contains different ordinate quantities; compare branch labels and purposes before interpreting overlays or a wide table."
        )
    return warnings


def build_postprocessing_export(
    workspace: BatchWorkspace,
    spectrum_ids: Sequence[str],
    *,
    branches: Sequence[str] = ("baseline",),
    valid_only: bool = False,
    include_wide: bool = False,
) -> BatchExportArtifact:
    """Capture confirmed nodes and serialize their complete B/S parent graph."""

    ready, report = _select(workspace, spectrum_ids, branches, valid_only)
    first_axis = branch_arrays(ready[0][2])[0]
    same_axis = all(
        np.array_equal(first_axis, branch_arrays(snapshot)[0]) for _, _, snapshot in ready
    )
    if include_wide and not same_axis:
        raise BatchError("AXIS_MISMATCH_FOR_WIDE_EXPORT", _AXIS_REASON)
    captured: dict[tuple[str, str], Snapshot] = {}

    def include(sid: str, branch: str, snapshot: Snapshot) -> None:
        key = sid, branch
        if key in captured:
            if captured[key].fingerprint != snapshot.fingerprint:
                raise BatchError(
                    "EXPORT_NOT_READY", "requested branches disagree on a parent version"
                )
            return
        if isinstance(snapshot, PostprocessSnapshot):
            include(sid, "baseline", snapshot.baseline)
            if snapshot.parent_smoothed is not None:
                include(sid, "smoothed", snapshot.parent_smoothed)
        captured[key] = snapshot

    for sid, branch, snapshot in ready:
        include(sid, branch, snapshot)
    members: dict[str, bytes] = {}
    nodes: list[dict[str, Any]] = []
    requested_keys = {(sid, branch) for sid, branch, _ in ready}
    for (sid, branch), snapshot in captured.items():
        node, added = _serialize_node(workspace, sid, branch, snapshot)
        if set(members) & set(added):
            raise BatchError("EXPORT_NOT_READY", "stable export member names collide")
        nodes.append(node)
        members.update(added)
    index = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": "1.0",
        "workspace_id": workspace.workspace_id,
        "nodes": [
            {
                "spectrum_id": node["spectrum_id"],
                "branch": node["branch"],
                "fingerprint": node["fingerprint"],
                "recipe_json": node["recipe_json"],
                "spectra_csv": node["spectra_csv"],
                "included_as": "requested"
                if (node["spectrum_id"], node["branch"]) in requested_keys
                else "parent",
            }
            for node in nodes
        ],
    }
    warnings = _heterogeneous_warnings(nodes)
    wide = None
    wide_headers: list[str] = []
    wide_path = "requested_branches_wide.csv" if include_wide else None
    if include_wide:
        for sid, branch, snapshot in ready:
            quantity = "absorbance" if isinstance(snapshot, StageSnapshot) else snapshot.quantity
            wide_headers.append(
                f"{_safe_name(workspace.records[sid].display_name)}__{sid}__{branch}__{_quantity_header(branch, quantity)}"
            )
        wide = _table(
            ["wavenumber_cm-1", *wide_headers],
            list(
                zip(
                    first_axis,
                    *(branch_arrays(snapshot)[1][0] for _, _, snapshot in ready),
                    strict=True,
                )
            ),
        )
        members[str(wide_path)] = wide
    metadata = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": "1.0",
        "workflow_mode": "independent_batch",
        "workspace_id": workspace.workspace_id,
        "is_2d_ready": False,
        "requested_spectrum_ids": list(dict.fromkeys(spectrum_ids)),
        "requested_branches": list(dict.fromkeys(branches)),
        "exported_items": [
            {"spectrum_id": sid, "branch": branch, "fingerprint": snapshot.fingerprint}
            for sid, branch, snapshot in ready
        ],
        "excluded_items": [row for row in report if row["status"] != "exported"],
        "selection_policy": "valid_only" if valid_only else "strict",
        "report": report,
        "warnings": warnings,
        "import_issues": plain(workspace.import_issues),
        "branch_index": "branch_index.json",
        "wide_csv": wide_path,
        "wide_headers": wide_headers,
        "wide_available": same_axis,
        "wide_unavailable_reason": None if same_axis else _AXIS_REASON,
        "baseline_verification_scope": _BASELINE_NOTICE,
        "baseline_version_scope": _BASELINE_VERSION_SCOPE,
        "exporter_versions": {
            **dependency_versions(),
            **{name: importlib.metadata.version(name) for name in ("pybaselines", "pydantic")},
        },
        "qc_applicability": {"cross_sample_time_continuity": "not_applicable"},
    }
    headers = sorted({key for row in report for key in row})
    members["processing_report.csv"] = _table(
        headers,
        [
            [
                json_bytes(row.get(key)).decode()
                if isinstance(row.get(key), (list, dict))
                else row.get(key, "")
                for key in headers
            ]
            for row in report
        ],
    )
    members["branch_index.json"] = json_bytes(index)
    members["batch_metadata.json"] = json_bytes(metadata)
    return BatchExportArtifact(
        zip_bytes=make_archive(members, ARTIFACT_TYPE),
        filename=f"ordinary_postprocessing_{_safe_name(workspace.workspace_id)[:12]}.zip",
        metadata=metadata,
        report=report,
        wide_csv_bytes=wide,
        wide_unavailable_reason=None if same_axis else _AXIS_REASON,
    )


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise BatchError("POSTPROCESS_EXPORT_INTEGRITY_FAILED", message)


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _csv_arrays(payload: bytes, headers: Sequence[str]) -> tuple[FloatArray, FloatArray]:
    stream = io.StringIO(payload.decode("utf-8"))
    header = next(csv.reader([stream.readline()]))
    expected = next(csv.reader(io.StringIO(_table(headers, []).decode())))
    _require(header == expected, "CSV quantity/header differs from its branch")
    data = np.loadtxt(stream, delimiter=",", ndmin=2)
    _require(
        data.ndim == 2 and data.shape[1] == len(headers) and data.shape[0] >= 2, "invalid CSV shape"
    )
    x, y = validate_spectral_arrays(data[:, 0], data[:, 1:].T)
    _require(
        _table(headers, list(zip(x, *y, strict=True))) == payload,
        "CSV is not the canonical lossless array serialization",
    )
    return x, y


def _verify_source(node: dict[str, Any]) -> None:
    source = node["source"]
    _require(isinstance(source, dict), "invalid source metadata")
    for field in (
        "spectrum_id",
        "source_id",
        "display_name",
        "original_filename",
        "source_column_label",
    ):
        _require(isinstance(source[field], str), "source identity/label must be text")
    _require(
        source["spectrum_id"] == node["spectrum_id"] and source["source_id"] == node["source_id"],
        "source/node identity mismatch",
    )
    _require(
        type(source["source_column_index"]) is int and source["source_column_index"] >= 1,
        "invalid intensity column index",
    )
    _require(
        _sha(source["source_sha256"]) and isinstance(source["import_options"], dict),
        "invalid source hash/options",
    )
    selection = json_fingerprint(
        {
            "source_sha256": source["source_sha256"],
            "column_index": source["source_column_index"],
            "import_options": source["import_options"],
        }
    )
    _require(selection == source["source_selection_sha256"], "source selection hash mismatch")
    _require(
        source["original_axis_direction"] in {"ascending", "descending"}
        and source["original_axis_direction"] == node["output_axis_direction"],
        "original/output axis direction mismatch",
    )
    _require(
        type(source["original_point_count"]) is int
        and source["original_point_count"] >= node["point_count"],
        "invalid original sample count",
    )


def _verify_baseline(node: dict[str, Any]) -> None:
    source, request = node["source"], node["stage_request"]
    config = PipelineConfig(**node["recipe"])
    require_independent_config(config)
    _require(
        node["effective_recipe"] == node["recipe"] == plain(config.to_dict()),
        "baseline configuration mismatch",
    )
    expected = {
        "workspace_id": node["workspace_id"],
        "spectrum_id": node["spectrum_id"],
        "source_id": node["source_id"],
        "original_filename": source["original_filename"],
        "original_column_label": source["source_column_label"],
        "source_selection_sha256": source["source_selection_sha256"],
        "scientific_input_sha256": node["input_sha256"],
        "workflow_mode": "independent_batch",
        "stage": "fine",
        "config": node["recipe"],
        "parent_coarse_fingerprint": request["parent_coarse_fingerprint"],
        "implementation": node["implementation"],
    }
    _require(expected == request, "baseline stage request identity/config mismatch")
    _require(
        json_fingerprint(request)
        == node["fingerprint"]
        == node["request_fingerprint"]
        == node["baseline_fingerprint"],
        "baseline stage fingerprint mismatch",
    )
    coarse_request = {
        **request,
        "stage": "coarse",
        "parent_coarse_fingerprint": None,
        "config": coarse_config(preparation_from_config(config), config.coarse_baseline).to_dict(),
        "implementation": node["parent_coarse_implementation"],
    }
    _require(
        _sha(node["parent_coarse_implementation"])
        and json_fingerprint(coarse_request) == request["parent_coarse_fingerprint"],
        "baseline coarse-parent anchor mismatch",
    )
    _require(
        node["quantity"] == "absorbance" and node["purpose"] == "scientific",
        "baseline quantity mismatch",
    )
    _require(
        node["parent_fingerprint"] is None and node["parent_branch"] is None,
        "baseline has an unexpected postprocessing parent",
    )
    _require(
        node["fine_decision"] in {"applied", "explicitly_skipped"},
        "baseline lacks a final decision",
    )
    if node["fine_decision"] == "explicitly_skipped":
        _require(
            not config.fine_baseline.enabled, "explicitly skipped baseline recipe enables fine"
        )
    _require(
        node["removed_csv"] is None and node["scale"] is None and node["offset"] is None,
        "baseline carries unexpected derived components",
    )
    _require(
        node["verification_scope"] == _BASELINE_NOTICE, "baseline verification scope is missing"
    )
    _require(
        node["version_scope"] == _BASELINE_VERSION_SCOPE
        and isinstance(node["versions"], dict)
        and set(node["versions"]) == {"ftir_baseline"}
        and isinstance(node["versions"]["ftir_baseline"], str)
        and bool(node["versions"]["ftir_baseline"]),
        "baseline software version/scope is missing",
    )
    conversion = node["unit_conversion"]
    formulas = {
        "absorbance": "A = absorbance (identity copy)",
        "percent_transmittance": "A = -log10(percent_transmittance / 100)",
        "fraction_transmittance": "A = -log10(fraction_transmittance)",
    }
    _require(
        isinstance(conversion, dict)
        and conversion["input_unit"] == config.input_unit
        and conversion["output_unit"] == "absorbance"
        and conversion["formula"] == formulas[config.input_unit]
        and conversion["transmittance_floor"] == config.transmittance_floor,
        "baseline unit conversion record disagrees with its recipe",
    )
    positions = conversion["repaired_indices"]
    _require(
        type(conversion["repaired_count"]) is int
        and isinstance(positions, list)
        and conversion["repaired_count"] == len(positions),
        "invalid repair count",
    )
    _require(
        all(
            isinstance(position, list)
            and len(position) == 2
            and type(position[0]) is int
            and position[0] == 0
            and type(position[1]) is int
            and 0 <= position[1] < source["original_point_count"]
            for position in positions
        ),
        "invalid repair positions",
    )
    _require(
        len({tuple(position) for position in positions}) == len(positions),
        "duplicate repair positions",
    )
    _require(
        config.transmittance_floor is not None or not positions,
        "repairs require an explicit transmittance floor",
    )


def _derived_fingerprint(node: Mapping[str, Any], removed: FloatArray | None) -> str:
    return json_fingerprint(
        {
            "request": node["request_fingerprint"],
            "x": node["x_sha256"],
            "y": node["y_sha256"],
            "quantity": node["quantity"],
            "purpose": node["purpose"],
            "removed": None if removed is None else array_hash(removed),
            "scale": None
            if node["scale"] is None
            else array_hash(np.asarray(node["scale"], dtype=float)),
            "offset": None
            if node["offset"] is None
            else array_hash(np.asarray(node["offset"], dtype=float)),
            "reference_details": node["reference_details"],
            "qc": node["qc"],
            "warnings": node["warnings"],
            "versions": node["versions"],
        }
    )


def _verify_derived(
    node: dict[str, Any],
    parent: dict[str, Any],
    baseline: dict[str, Any],
    x: FloatArray,
    y: FloatArray,
    parent_y: FloatArray,
    removed: FloatArray | None,
    *,
    recompute: bool,
) -> None:
    config = effective_config(node["branch"], node["recipe"])
    _require(
        plain(config.scientific_dict()) == node["effective_recipe"], "effective recipe mismatch"
    )
    _require(
        node["parent_branch"] == PARENTS[node["branch"]]
        and node["parent_fingerprint"] == parent["fingerprint"],
        "derived parent identity mismatch",
    )
    _require(
        node["source"] == parent["source"] == baseline["source"],
        "parent/source provenance mismatch",
    )
    _require(
        node["baseline_fingerprint"] == baseline["fingerprint"] == parent["baseline_fingerprint"],
        "baseline anchor mismatch",
    )
    _require(node["input_sha256"] == baseline["input_sha256"], "raw scientific input hash mismatch")
    payload = {
        "contract": "ordinary-postprocess-v1",
        "workspace_id": node["workspace_id"],
        "spectrum_id": node["spectrum_id"],
        "source_id": node["source_id"],
        "input_sha256": node["input_sha256"],
        "branch": node["branch"],
        "baseline_fingerprint": baseline["fingerprint"],
        "parent_fingerprint": parent["fingerprint"],
        "parent_branch": PARENTS[node["branch"]],
        "x_sha256": array_hash(x),
        "parent_y_sha256": array_hash(parent_y),
        "effective_recipe": node["effective_recipe"],
        "implementation": node["implementation"],
    }
    _require(
        payload == node["request_payload"]
        and json_fingerprint(payload) == node["request_fingerprint"],
        "derived request fingerprint mismatch",
    )
    _require(
        isinstance(node["versions"], dict) and bool(node["versions"]), "missing numerical versions"
    )
    if node["branch"] == "smoothed":
        _require(
            node["quantity"] == "absorbance" and node["purpose"] == "scientific",
            "invalid smoothed quantity",
        )
        _require(
            removed is not None and np.array_equal(removed, parent_y - y),
            "smoothing residual identity mismatch",
        )
        _require(
            node["scale"] is None and node["offset"] is None,
            "smoothing carries normalization factors",
        )
        if recompute:
            # Bound untrusted replay work; these limits do not change the filter.
            full = config.to_dict()
            if config.method == "gaussian":
                _require(
                    full["gaussian_sigma_points"] * full["gaussian_truncate"] <= 1_000_000,
                    "Gaussian replay kernel exceeds verifier resource limit",
                )
            result = smooth_spectral_arrays(x, parent_y, full)
            _require(
                np.array_equal(result.smoothed_spectra, y),
                "smoothing replay differs from exported output",
            )
            expected_qc = {
                "per_spectrum": plain(result.per_spectrum_metrics),
                "summary": plain(result.summary_metrics),
                "median_wavenumber_spacing": result.median_wavenumber_spacing,
                "spacing_relative_max_deviation": result.spacing_relative_max_deviation,
                "approximate_physical_width": plain(result.approximate_physical_width),
            }
            _require(expected_qc == node["qc"], "smoothing QC differs from replay")
            _require(
                list(result.warnings) == node["warnings"], "smoothing warnings differ from replay"
            )
    else:
        minmax = config.method == "minmax_display"
        _require(
            node["quantity"] == ("minmax_scaled_intensity" if minmax else "normalized_intensity"),
            "normalization quantity mismatch",
        )
        _require(
            node["purpose"] == ("display_only" if minmax else "scientific") and removed is None,
            "normalization purpose/components mismatch",
        )
        scale, offset = (
            np.asarray(node["scale"], dtype=float),
            np.asarray(node["offset"], dtype=float),
        )
        _require(
            scale.shape == offset.shape == (1,)
            and np.isfinite(scale).all()
            and np.isfinite(offset).all()
            and np.all(scale > 0),
            "invalid finite affine coefficients",
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if minmax:
                minimum = np.min(parent_y, axis=1)
                span = np.max(parent_y, axis=1) - minimum
                _require(
                    np.array_equal(scale, 1 / span) and np.array_equal(offset, -minimum / span),
                    "Min-Max scale/offset mismatch",
                )
                expected = (parent_y - minimum[:, None]) * scale[:, None]
            else:
                _require(np.all(offset == 0), "scientific normalization must not add an offset")
                expected = parent_y * scale[:, None]
        _require(np.array_equal(expected, y), "normalization affine identity mismatch")
        if recompute:
            normalized = normalize_spectral_arrays(x, parent_y, config.to_dict())
            _require(
                np.array_equal(normalized.normalized_spectra, y)
                and np.array_equal(normalized.scale, scale)
                and np.array_equal(normalized.offset, offset),
                "normalization replay differs from exported output",
            )
            _require(
                plain(normalized.reference_details) == node["reference_details"]
                and plain(normalized.metrics) == node["qc"]
                and list(normalized.warnings) == node["warnings"],
                "normalization reference/QC evidence differs from replay",
            )
    _require(
        _derived_fingerprint(node, removed) == node["fingerprint"],
        "derived output fingerprint mismatch",
    )


def verify_postprocessing_export(payload: bytes, *, recompute: bool = True) -> bool:
    """Check a bundle; optionally replay only S/N using its exported parents.

    Baseline provenance hashes and recipes are integrity anchors, not a baseline
    recomputation or source authentication. Replay uses installed numerical code;
    changed dependencies may produce a verification failure. No source bytes,
    Python objects, Prepared datasets or pipeline executions are loaded.
    """
    try:
        _require(type(recompute) is bool, "recompute must be bool")
        members = read_verified_archive(payload, ARTIFACT_TYPE)
        metadata, index = (
            _read_json(members["batch_metadata.json"]),
            _read_json(members["branch_index.json"]),
        )
        for document in (metadata, index):
            _require(
                document["artifact_type"] == ARTIFACT_TYPE and document["schema_version"] == "1.0",
                "unsupported bundle schema",
            )
            _require(
                isinstance(document["workspace_id"], str) and bool(document["workspace_id"]),
                "invalid workspace identity",
            )
        _require(
            metadata["workspace_id"] == index["workspace_id"] and metadata["is_2d_ready"] is False,
            "workspace or 2D applicability mismatch",
        )
        _require(
            metadata["baseline_verification_scope"] == _BASELINE_NOTICE
            and metadata["branch_index"] == "branch_index.json",
            "missing verification scope/index",
        )
        _require(
            metadata["baseline_version_scope"] == _BASELINE_VERSION_SCOPE,
            "missing baseline version scope",
        )
        versions = metadata["exporter_versions"]
        _require(
            isinstance(versions, dict)
            and set(versions)
            == {"python", "numpy", "scipy", "ftir-spectral-workbench", "pybaselines", "pydantic"}
            and all(isinstance(version, str) and bool(version) for version in versions.values()),
            "missing export environment versions",
        )
        _require(
            metadata["selection_policy"] in {"strict", "valid_only"}, "invalid selection policy"
        )
        ids, branches = metadata["requested_spectrum_ids"], metadata["requested_branches"]
        _require(
            isinstance(ids, list)
            and bool(ids)
            and all(isinstance(sid, str) and sid for sid in ids)
            and len(ids) == len(set(ids)),
            "invalid requested identities",
        )
        _require(
            isinstance(branches, list)
            and bool(branches)
            and all(branch in BRANCHES for branch in branches)
            and len(branches) == len(set(branches)),
            "invalid requested branches",
        )
        report = metadata["report"]
        _require(isinstance(report, list), "invalid report")
        _require(
            [(row["spectrum_id"], row["branch"]) for row in report]
            == [(sid, branch) for sid in ids for branch in branches],
            "report does not cover exactly every requested item",
        )
        exported = [row for row in report if row["status"] == "exported"]
        excluded = [row for row in report if row["status"] != "exported"]
        _require(
            bool(exported) and excluded == metadata["excluded_items"], "excluded report mismatch"
        )
        _require(
            metadata["selection_policy"] != "strict" or not excluded,
            "strict bundle excludes requested branches",
        )
        _require(
            all(
                row["status"] in {"exported", "missing", "stale", "failed", "excluded"}
                for row in report
            ),
            "invalid report status",
        )
        _require(
            metadata["exported_items"]
            == [
                {
                    "spectrum_id": row["spectrum_id"],
                    "branch": row["branch"],
                    "fingerprint": row["fingerprint"],
                }
                for row in exported
            ],
            "exported index/report mismatch",
        )
        report_headers = sorted({key for row in report for key in row})
        expected_report = _table(
            report_headers,
            [
                [
                    json_bytes(row.get(key)).decode()
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key, "")
                    for key in report_headers
                ]
                for row in report
            ],
        )
        _require(
            members["processing_report.csv"] == expected_report, "CSV report differs from metadata"
        )
        nodes: dict[tuple[str, str], dict[str, Any]] = {}
        arrays: dict[tuple[str, str], tuple[FloatArray, FloatArray]] = {}
        used = {"batch_metadata.json", "branch_index.json", "processing_report.csv"}
        index_nodes = index["nodes"]
        _require(
            isinstance(index_nodes, list) and len(index_nodes) <= len(ids) * len(BRANCHES),
            "invalid branch index size",
        )
        requested = {(row["spectrum_id"], row["branch"]) for row in exported}
        for entry in index_nodes:
            key = entry["spectrum_id"], entry["branch"]
            _require(
                key not in nodes and key[0] in ids and key[1] in BRANCHES,
                "duplicate/unknown branch identity",
            )
            node = _read_json(members[entry["recipe_json"]])
            _require(
                isinstance(node, dict) and node["schema_version"] == "1.0", "invalid node schema"
            )
            integrity = node["node_integrity_sha256"]
            _require(
                integrity
                == json_fingerprint(
                    {k: v for k, v in node.items() if k != "node_integrity_sha256"}
                ),
                "node metadata integrity mismatch",
            )
            _require(
                node["workspace_id"] == metadata["workspace_id"]
                and (node["spectrum_id"], node["branch"]) == key,
                "node/index/workspace mismatch",
            )
            _require(
                node["fingerprint"] == entry["fingerprint"]
                and node["recipe_json"] == entry["recipe_json"]
                and node["spectra_csv"] == entry["spectra_csv"],
                "node file/fingerprint index mismatch",
            )
            _require(
                entry["included_as"] == ("requested" if key in requested else "parent"),
                "implicit branch inclusion mismatch",
            )
            _require(
                node["is_2d_ready"] is False
                and all(
                    _sha(node[field])
                    for field in (
                        "input_sha256",
                        "fingerprint",
                        "baseline_fingerprint",
                        "request_fingerprint",
                        "implementation",
                        "x_sha256",
                        "y_sha256",
                    )
                ),
                "invalid scientific digest/2D flag",
            )
            _verify_source(node)
            _require(
                _node_paths(node["source"], node["branch"], node["quantity"])
                == (node["spectra_csv"], node["recipe_json"], node["removed_csv"]),
                "branch filename identity/quantity mismatch",
            )
            _require(
                node["column_label"] == _quantity_header(node["branch"], node["quantity"]),
                "branch column quantity mismatch",
            )
            _require(
                isinstance(node["warnings"], list)
                and all(isinstance(value, str) for value in node["warnings"]),
                "invalid warning metadata",
            )
            x, y = _csv_arrays(
                members[node["spectra_csv"]], ["wavenumber_cm-1", node["column_label"]]
            )
            _require(
                y.shape[0] == 1
                and type(node["point_count"]) is int
                and node["point_count"] == x.size,
                "invalid spectrum point/row count",
            )
            _require(
                array_hash(x) == node["x_sha256"] and array_hash(y) == node["y_sha256"],
                "array hash mismatch",
            )
            _require(
                node["output_axis_direction"] == ("ascending" if x[0] < x[-1] else "descending"),
                "axis direction mismatch",
            )
            if node["branch"] == "baseline":
                _verify_baseline(node)
            nodes[key], arrays[key] = node, (x, y)
            _require(
                node["spectra_csv"] not in used and node["recipe_json"] not in used,
                "duplicate branch member reference",
            )
            used.update((node["spectra_csv"], node["recipe_json"]))
        required: set[tuple[str, str]] = set()
        for row in exported:
            key = row["spectrum_id"], row["branch"]
            _require(
                key in nodes and nodes[key]["fingerprint"] == row["fingerprint"],
                "requested branch absent or changed",
            )
            required.add(key)
            required.add((key[0], "baseline"))
            if key[1] == "normalized_smoothed":
                required.add((key[0], "smoothed"))
            source = nodes[key]["source"]
            _require(
                all(row[field] == source[field] for field in source),
                "report/source identity mismatch",
            )
        _require(set(nodes) == required, "graph is missing parents or contains unrequested extras")
        for branch in BRANCHES[1:]:
            for key, node in nodes.items():
                if key[1] != branch:
                    continue
                parent_key = key[0], PARENTS[branch]
                parent = nodes[parent_key]
                x, y = arrays[key]
                parent_x, parent_y = arrays[parent_key]
                _require(np.array_equal(x, parent_x), "derived/parent axes differ")
                removed = None
                if node["removed_csv"] is not None:
                    removed_x, removed = _csv_arrays(
                        members[node["removed_csv"]],
                        ["wavenumber_cm-1", "smoothing_removed_component"],
                    )
                    _require(
                        np.array_equal(removed_x, x)
                        and removed.shape == y.shape
                        and array_hash(removed) == node["removed_sha256"],
                        "removed component hash/axis mismatch",
                    )
                    _require(
                        node["removed_csv"] not in used, "duplicate removed component reference"
                    )
                    used.add(node["removed_csv"])
                else:
                    _require(node["removed_sha256"] is None, "unexpected removed hash")
                _verify_derived(
                    node,
                    parent,
                    nodes[(key[0], "baseline")],
                    x,
                    y,
                    parent_y,
                    removed,
                    recompute=recompute,
                )
        _require(
            metadata["warnings"] == _heterogeneous_warnings(list(nodes.values())),
            "heterogeneous comparison warnings mismatch",
        )
        ordered = [arrays[(row["spectrum_id"], row["branch"])] for row in exported]
        same_axis = all(np.array_equal(ordered[0][0], pair[0]) for pair in ordered)
        _require(
            metadata["wide_available"] is same_axis
            and metadata["wide_unavailable_reason"] == (None if same_axis else _AXIS_REASON),
            "wide availability does not match actual output coordinates",
        )
        if metadata["wide_csv"] is not None:
            _require(
                metadata["wide_csv"] == "requested_branches_wide.csv" and same_axis,
                "invalid wide export",
            )
            headers = [
                f"{_safe_name(nodes[(row['spectrum_id'], row['branch'])]['source']['display_name'])}__{row['spectrum_id']}__{row['branch']}__{nodes[(row['spectrum_id'], row['branch'])]['column_label']}"
                for row in exported
            ]
            _require(
                headers == metadata["wide_headers"],
                "wide labels do not identify requested branches",
            )
            wx, wy = _csv_arrays(members[metadata["wide_csv"]], ["wavenumber_cm-1", *headers])
            _require(
                np.array_equal(wx, ordered[0][0])
                and np.array_equal(wy, np.vstack([pair[1][0] for pair in ordered])),
                "wide array values differ from confirmed per-spectrum arrays",
            )
            used.add(metadata["wide_csv"])
        else:
            _require(metadata["wide_headers"] == [], "unexpected wide headers")
        _require(used == set(members), "unexpected or unreferenced bundle members")
        return True
    except (
        BatchError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        StopIteration,
        OSError,
        csv.Error,
        FloatingPointError,
        OverflowError,
        RecursionError,
    ):
        return False


__all__ = [
    "ARTIFACT_TYPE",
    "build_postprocessing_export",
    "can_export_postprocessing_wide",
    "verify_postprocessing_export",
]
