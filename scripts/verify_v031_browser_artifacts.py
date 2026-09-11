#!/usr/bin/env python3
"""Audit actual v0.3.1 browser downloads; never create replacement evidence.

Run after the browser has downloaded all required files. ``--partial`` allows
development against an incomplete directory but reports PARTIAL, never PASS.
Replay is explicit and confined to the bundle verifier's smoothing/normalization
checks. Workspace loading and non-replay verification forbid numerical work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from ftir_workbench.batch import postprocessing_export as export_module
from ftir_workbench.batch.fingerprints import source_selection_fingerprint
from ftir_workbench.batch.models import BatchWorkspace, StageSnapshot
from ftir_workbench.batch.postprocessing import (
    BRANCHES,
    DERIVED_BRANCHES,
    PostprocessSnapshot,
    branch_arrays,
    get_branch,
    plain,
)
from ftir_workbench.batch.workspace import load_batch_workspace, read_verified_archive

REQUIRED_FILES = (
    "all_branches.zip", "workspace_before.zip", "workspace_restored.zip",
    "all_branches_restored.zip", "selected_wide.zip", "selected_wide.csv",
    "stale_valid_only.zip",
)
BASELINE_AND_PREPARED = (
    "ftir_baseline.pipeline.run_pipeline",
    "ftir_workbench.batch.service.run_pipeline",
    "ftir_workbench.services.baseline_service.BaselineWorkflowService.run",
    "ftir_workbench.models.PreparedSpectralDataset.__init__",
    "ftir_workbench.adapters.prepared_from_baseline_result",
    "ftir_workbench.adapters.prepared_scientific_branch_from_baseline_result",
    "ftir_workbench.adapters.prepared_from_smoothed_result",
    "ftir_workbench.services.twodcos_service.TwoDCOSWorkflowService.compute",
)
NUMERICAL_ENTRYPOINTS = (
    "ftir_workbench.post_baseline_smoothing.apply_post_baseline_smoothing",
    "ftir_workbench.post_baseline_smoothing.smooth_spectral_arrays",
    "ftir_workbench.batch.postprocessing.smooth_spectral_arrays",
    "ftir_workbench.batch.postprocessing.normalize_spectral_arrays",
    "ftir_workbench.batch.normalization_adapter.normalize_spectral_arrays",
    "ftir_workbench.batch.normalization_adapter.apply_normalization",
    "ftir_baseline.normalization.apply_normalization",
    "ftir_baseline.normalization.normalize_spectra",
    "ftir_workbench.batch.postprocessing_export.smooth_spectral_arrays",
    "ftir_workbench.batch.postprocessing_export.normalize_spectral_arrays",
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextmanager
def forbid_calls(*, numerical: bool) -> Iterator[Counter[str]]:
    """Fail even when a caller catches the spy exception and returns False."""
    calls: Counter[str] = Counter()

    def forbidden(name: str) -> Callable[..., None]:
        def fail(*args: Any, **kwargs: Any) -> None:
            calls[name] += 1
            raise AssertionError("Forbidden audit-time calculation: " + name)
        return fail

    with ExitStack() as stack:
        for name in BASELINE_AND_PREPARED + (NUMERICAL_ENTRYPOINTS if numerical else ()):
            stack.enter_context(patch(name, side_effect=forbidden(name)))
        yield calls
        require(not calls, f"Forbidden audit calls observed: {dict(calls)}")


def csv_values(payload: bytes) -> tuple[list[str], np.ndarray]:
    stream = io.StringIO(payload.decode("utf-8"))
    header = next(csv.reader([stream.readline()]))
    values = np.loadtxt(stream, delimiter=",", dtype=np.float64, ndmin=2)
    require(values.ndim == 2 and values.shape[1] == len(header), "CSV shape/header mismatch")
    require(np.all(np.isfinite(values)), "CSV contains nonfinite values")
    return header, values


def inspect_export(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = path.read_bytes()
    with forbid_calls(numerical=True) as no_replay_calls:
        require(export_module.verify_postprocessing_export(payload, recompute=False),
                path.name + ": integrity/non-replay verifier failed")
    replay_calls: Counter[str] = Counter()
    smooth = export_module.smooth_spectral_arrays
    normalize = export_module.normalize_spectral_arrays

    def replay_smooth(*args: Any, **kwargs: Any) -> Any:
        replay_calls["smoothing"] += 1
        return smooth(*args, **kwargs)

    def replay_normalize(*args: Any, **kwargs: Any) -> Any:
        replay_calls["normalization"] += 1
        return normalize(*args, **kwargs)

    with (
        forbid_calls(numerical=False) as forbidden_replay_calls,
        patch.object(export_module, "smooth_spectral_arrays", side_effect=replay_smooth),
        patch.object(export_module, "normalize_spectral_arrays", side_effect=replay_normalize),
    ):
        require(export_module.verify_postprocessing_export(payload, recompute=True),
                path.name + ": explicit S/N replay verifier failed")
    members = read_verified_archive(payload, export_module.ARTIFACT_TYPE)
    metadata = json.loads(members["batch_metadata.json"])
    index = json.loads(members["branch_index.json"])
    nodes = {}
    for item in index["nodes"]:
        node = json.loads(members[item["recipe_json"]])
        header, values = csv_values(members[item["spectra_csv"]])
        require(header == ["wavenumber_cm-1", node["column_label"]], "Wrong quantity column")
        key = item["spectrum_id"], item["branch"]
        require(key not in nodes, "Duplicate branch node")
        nodes[key] = {"node": node, "values": values, "included_as": item["included_as"]}
    expected = Counter({"smoothing": sum(branch == "smoothed" for _, branch in nodes),
                        "normalization": sum(branch.startswith("normalized_") for _, branch in nodes)})
    require(replay_calls == expected, f"Unexpected explicit verifier replay counts: {dict(replay_calls)} != {dict(expected)}")
    require(metadata["is_2d_ready"] is False, "Ordinary export claims 2D readiness")
    require(not any("for_2dcos" in name for name in members), "Ordinary export includes for_2dcos")
    summary = {
        "filename": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload), "nodes": len(nodes),
        "requested_items": len(metadata["exported_items"]),
        "excluded_items": metadata["excluded_items"],
        "integrity_verifier": True, "explicit_replay_verifier": True,
        "non_replay_forbidden_calls": sum(no_replay_calls.values()),
        "replay_baseline_prepared_2d_calls": sum(forbidden_replay_calls.values()),
        "verification_only_replays": dict(replay_calls),
        "exporter_versions": metadata["exporter_versions"],
        "wide_available": metadata["wide_available"],
    }
    return {"metadata": metadata, "nodes": nodes, "members": members}, summary


def assert_equal(left: Any, right: Any, label: str, counts: Counter[str]) -> None:
    """Compare complete immutable parents, including all PipelineResult arrays."""
    if isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray), label + ": array type changed")
        require(left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True),
                label + ": array contents, shape or dtype changed")
        require(not right.flags.writeable, label + ": restored scientific array is mutable")
        counts["arrays"] += 1
        counts["array_elements"] += left.size
    elif isinstance(left, Mapping):
        require(isinstance(right, Mapping) and left.keys() == right.keys(), label + ": mapping keys changed")
        for key in left:
            assert_equal(left[key], right[key], f"{label}.{key}", counts)
    elif is_dataclass(left) and not isinstance(left, type):
        require(type(left) is type(right), label + ": dataclass type changed")
        for field in fields(left):
            assert_equal(getattr(left, field.name), getattr(right, field.name), label + "." + field.name, counts)
    elif isinstance(left, (list, tuple)):
        require(isinstance(right, (list, tuple)) and len(left) == len(right), label + ": sequence changed")
        for index, (item, other) in enumerate(zip(left, right, strict=True)):
            assert_equal(item, other, f"{label}[{index}]", counts)
    elif hasattr(left, "to_dict"):
        assert_equal(left.to_dict(), right.to_dict(), label, counts)
    elif isinstance(left, float) and math.isnan(left):
        require(isinstance(right, float) and math.isnan(right), label + ": diagnostic NaN changed")
    else:
        require(left == right, label + ": value changed")


def inspect_workspace(path: Path, input_directory: Path) -> tuple[BatchWorkspace, dict[str, Any]]:
    payload = path.read_bytes()
    with forbid_calls(numerical=True) as calls:
        workspace = load_batch_workspace(payload)
    for source in workspace.sources.values():
        require(Path(source.original_filename).name == source.original_filename,
                "Browser synthetic input unexpectedly contains a path")
        original_path = input_directory / source.original_filename
        require(original_path.is_file(), "Actual uploaded input is missing: " + source.original_filename)
        require(original_path.read_bytes() == source.original_bytes,
                "Workspace original source bytes differ from the actual uploaded file")
    for state in workspace.states.values():
        require(state.coarse_preview is None and state.fine_preview is None,
                "Workspace restored a baseline preview")
    for state in workspace.postprocessing.values():
        require(all(getattr(state, branch).preview is None for branch in DERIVED_BRANCHES),
                "Workspace restored a postprocessing preview")
    return workspace, {
        "filename": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload), "workspace_id": workspace.workspace_id,
        "spectrum_ids": list(workspace.display_order), "source_count": len(workspace.sources),
        "selected_spectrum_id": workspace.selected_spectrum_id,
        "selected_spectrum_ids": workspace.selected_spectrum_ids,
        "load_numerical_baseline_prepared_2d_calls": sum(calls.values()),
        "source_bytes_match_actual_uploads": True, "previews_absent": True,
    }


def compare_workspaces(before: BatchWorkspace, after: BatchWorkspace) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for name in ("schema_version", "workspace_id", "workflow_mode", "sources", "records",
                 "display_order", "selected_spectrum_id", "selected_spectrum_ids", "import_issues"):
        assert_equal(getattr(before, name), getattr(after, name), name, counts)
    require(before.states.keys() == after.states.keys(), "Baseline processing IDs changed")
    assert_equal(before.states, after.states, "baseline_states", counts)
    assert_equal(before.postprocessing, after.postprocessing, "postprocessing", counts)
    return {
        "stable_ids_sources_drafts_arrays_and_parents_equal": True,
        "selection_source_export_and_display_preferences_equal": True,
        "compared_arrays": counts["arrays"], "compared_array_elements": counts["array_elements"],
        "export_history_equal": before.export_history == after.export_history,
        "note": "Download/export activity history is reported separately from restored scientific state.",
    }


def compare_export_workspace(bundle: dict[str, Any], workspace: BatchWorkspace) -> dict[str, Any]:
    require(bundle["metadata"]["workspace_id"] == workspace.workspace_id, "Export/workspace identity differs")
    compared = 0
    nb_differences = []
    with forbid_calls(numerical=True):
        for (sid, branch), content in bundle["nodes"].items():
            node, values = content["node"], content["values"]
            snapshot = get_branch(workspace, sid, branch)
            x, y = branch_arrays(snapshot)
            require(np.array_equal(values[:, 0], x) and np.array_equal(values[:, 1], y[0]),
                    f"Downloaded CSV is not the confirmed float64 array: {sid}/{branch}")
            require(snapshot.fingerprint == node["fingerprint"], "Export fingerprint differs from workspace")
            record = workspace.records[sid]
            source = workspace.sources[record.source_id]
            provenance = node["source"]
            require(provenance["source_id"] == record.source_id
                    and provenance["original_filename"] == source.original_filename
                    and provenance["source_column_index"] == record.original_column_index
                    and provenance["source_column_label"] == record.original_column_label
                    and provenance["source_sha256"] == source.original_bytes_sha256
                    and provenance["source_selection_sha256"] == source_selection_fingerprint(workspace, sid),
                    "Downloaded provenance does not match its original source/column")
            if isinstance(snapshot, PostprocessSnapshot):
                require(snapshot.parent_fingerprint == node["parent_fingerprint"], "Export parent differs")
                require(plain(snapshot.recipe) == node["recipe"], "Export recipe differs")
                if snapshot.removed_component is not None:
                    _, removed = csv_values(bundle["members"][node["removed_csv"]])
                    require(np.array_equal(removed[:, 1], snapshot.removed_component[0]), "Removed component differs")
                if branch == "normalized_baseline":
                    parent_analysis = snapshot.baseline.result.analysis_data[0]
                    require(not np.array_equal(values[:, 1], parent_analysis),
                            "This browser normalization case did not demonstrate a transformed N_B output")
                    nb_differences.append({"spectrum_id": sid, "method": snapshot.recipe["method"],
                                           "differs_from_parent_analysis_data": True})
            else:
                require(isinstance(snapshot, StageSnapshot), "Unknown baseline snapshot type")
                require(plain(snapshot.config.to_dict()) == node["recipe"], "Baseline export recipe differs")
            compared += 1
    return {"csv_arrays_match_confirmed_workspace": compared, "normalized_baseline_evidence": nb_differences}


def compare_all_exports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    require(before["nodes"].keys() == after["nodes"].keys(), "Restored export branch set changed")
    for key, original in before["nodes"].items():
        restored = after["nodes"][key]
        require(np.array_equal(original["values"], restored["values"]), "Restored export arrays changed")
        require(original["node"] == restored["node"], "Restored export recipe/provenance changed")
    return {"branch_arrays_recipes_sources_and_fingerprints_equal": True, "nodes": len(before["nodes"])}


def check_all_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    ids = metadata["requested_spectrum_ids"]
    require(len(ids) == 3 and set(metadata["requested_branches"]) == set(BRANCHES),
            "Expected the actual three-spectrum/four-branch browser acceptance download")
    require(set(bundle["nodes"]) == {(sid, branch) for sid in ids for branch in BRANCHES},
            "All-branch download is incomplete")
    require(not metadata["excluded_items"], "Initial/restored all-branch bundle unexpectedly excludes items")
    require(metadata["wide_available"] is False and metadata["wide_csv"] is None,
            "Heterogeneous axes incorrectly produced a wide table")
    axes = [bundle["nodes"][(sid, "baseline")]["values"][:, 0] for sid in ids]
    require(any(not np.array_equal(axes[0], x) for x in axes[1:]), "Acceptance data are not heterogeneous")
    require(any("Heterogeneous" in warning for warning in metadata["warnings"]), "Heterogeneous recipe warning absent")
    return {"three_spectra_twelve_branches": True, "heterogeneous_axes_disable_wide": True}


def check_selected_wide(bundle: dict[str, Any], csv_path: Path,
                        workspace: BatchWorkspace | None) -> dict[str, Any]:
    metadata = bundle["metadata"]
    ids = metadata["requested_spectrum_ids"]
    require(len(ids) == 2 and metadata["wide_available"] is True, "Selected wide export is not two same-axis spectra")
    require(metadata["wide_csv"] is not None, "Wide ZIP omits its requested wide CSV")
    downloaded = csv_path.read_bytes()
    require(bundle["members"][metadata["wide_csv"]] == downloaded, "Standalone wide CSV differs from the ZIP member")
    header, wide = csv_values(downloaded)
    require(header == ["wavenumber_cm-1", *metadata["wide_headers"]], "Standalone wide header differs")
    require(wide.shape[1] == len(metadata["exported_items"]) + 1, "Wrong wide table column count")
    for column, item in enumerate(metadata["exported_items"], start=1):
        values = bundle["nodes"][(item["spectrum_id"], item["branch"])]["values"]
        require(np.array_equal(wide[:, 0], values[:, 0]) and np.array_equal(wide[:, column], values[:, 1]),
                "Wide column is not an exact independently confirmed array")
    if workspace is not None:
        records = [workspace.records[sid] for sid in ids]
        require(records[0].source_id == records[1].source_id
                and {record.original_column_index for record in records} == {1, 2},
                "Selected wide spectra are not the actual A/B columns of the uploaded wide file")
    return {"selected_spectrum_ids": ids, "standalone_csv_equals_zip_member": True,
            "rows": wide.shape[0], "independent_output_columns": wide.shape[1] - 1,
            "all_columns_exact_float64_matches": True}


def check_stale(bundle: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    metadata = bundle["metadata"]
    require(metadata["selection_policy"] == "valid_only", "Stale acceptance download was not explicitly valid-only")
    require(metadata["requested_branches"] == ["normalized_smoothed"], "Stale acceptance export must explicitly request N_S")
    excluded = metadata["excluded_items"]
    require(excluded and any(row["status"] == "stale" for row in excluded), "No stale item was reported")
    excluded_ids = {row["spectrum_id"] for row in excluded}
    require(all(sid not in excluded_ids for sid, _ in bundle["nodes"]), "Stale N_S item entered the result graph")
    require(all(item["branch"] == "normalized_smoothed" for item in metadata["exported_items"]),
            "Stale branch export fell back to another branch")
    if reference is not None:
        for key, current in bundle["nodes"].items():
            require(key in reference["nodes"], "Unrelated new branch appeared in stale export")
            previous = reference["nodes"][key]
            require(current["node"]["fingerprint"] == previous["node"]["fingerprint"]
                    and np.array_equal(current["values"], previous["values"]),
                    "Changing excluded A changed an unrelated exported spectrum")
    return {"stale_exclusions": excluded, "no_fallback_or_stale_arrays": True,
            "unrelated_nodes_unchanged": reference is not None}


def audit(directory: Path, *, partial: bool) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    report: dict[str, Any] = {
        "started_utc": datetime.now(UTC).isoformat(), "mode": "partial" if partial else "complete",
        "missing_required_files": missing, "checks": {}, "failures": [],
        "scope": "Only actual browser download files were read; no spectra or replacement artifacts were generated.",
        "spy_scope": "Spies observe this audit's loading/verification calls, not past browser-server executions.",
        "integrity_notice": "Hashes establish internal integrity, not source authenticity or baseline refitting.",
    }
    bundles: dict[str, dict[str, Any]] = {}
    workspaces: dict[str, BatchWorkspace] = {}

    def run(name: str, action: Callable[[], Any]) -> Any:
        try:
            result = action()
            report["checks"][name] = {"status": "pass", "details": result}
            return result
        except Exception as exc:
            report["checks"][name] = {"status": "fail", "error": str(exc)}
            report["failures"].append({"check": name, "error": str(exc)})
            return None

    for name in REQUIRED_FILES:
        if name in missing or not name.endswith(".zip"):
            continue
        if name.startswith("workspace_"):
            def read_workspace(name: str = name) -> dict[str, Any]:
                ws, summary = inspect_workspace(directory / name, directory / "inputs")
                workspaces[name] = ws
                return summary
            run(name, read_workspace)
        else:
            def read_export(name: str = name) -> dict[str, Any]:
                bundle, summary = inspect_export(directory / name)
                bundles[name] = bundle
                return summary
            run(name, read_export)
    for name in ("all_branches.zip", "all_branches_restored.zip"):
        if name in bundles:
            run(name + ":complete_graph", lambda name=name: check_all_bundle(bundles[name]))
    before_ws, after_ws = workspaces.get("workspace_before.zip"), workspaces.get("workspace_restored.zip")
    if before_ws is not None and after_ws is not None:
        run("workspace_restore_equivalence", lambda: compare_workspaces(before_ws, after_ws))
    for export_name, workspace_name in (
        ("all_branches.zip", "workspace_before.zip"),
        ("all_branches_restored.zip", "workspace_restored.zip"),
        ("selected_wide.zip", "workspace_before.zip"),
        ("stale_valid_only.zip", "workspace_before.zip"),
    ):
        if export_name in bundles and workspace_name in workspaces:
            run(export_name + ":workspace_arrays", lambda en=export_name, wn=workspace_name:
                compare_export_workspace(bundles[en], workspaces[wn]))
    if "all_branches.zip" in bundles and "all_branches_restored.zip" in bundles:
        run("restored_export_equivalence", lambda: compare_all_exports(bundles["all_branches.zip"], bundles["all_branches_restored.zip"]))
    if "selected_wide.zip" in bundles and "selected_wide.csv" not in missing:
        run("selected_wide_exact_columns", lambda: check_selected_wide(bundles["selected_wide.zip"], directory / "selected_wide.csv", before_ws))
    if "stale_valid_only.zip" in bundles:
        run("stale_export_isolation", lambda: check_stale(bundles["stale_valid_only.zip"], bundles.get("all_branches.zip")))
    report["status"] = ("fail" if report["failures"] or (missing and not partial)
                        else "partial" if missing else "pass")
    report["completed_utc"] = datetime.now(UTC).isoformat()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/validation-private/v031/browser")
    parser.add_argument("--partial", action="store_true", help="Audit present real files; report missing evidence without claiming complete success")
    parser.add_argument("--output", type=Path, help="Optional audit-summary JSON output; never contains spectrum arrays")
    args = parser.parse_args()
    result = audit(args.directory, partial=args.partial)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
