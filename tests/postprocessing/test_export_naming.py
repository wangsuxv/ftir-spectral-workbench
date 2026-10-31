"""Readable postprocessing names preserve confirmed nodes and strict bundle graphs."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch.export import _table, build_batch_export, verify_batch_export
from ftir_workbench.batch.export_naming import (
    get_naming_options,
    make_name_row,
    name_map_table,
    set_name_override,
    set_naming_policy,
)
from ftir_workbench.batch.fingerprints import json_fingerprint
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchError, BatchWorkspace
from ftir_workbench.batch.postprocessing import BRANCHES, get_branch, update_draft
from ftir_workbench.batch.postprocessing_export import (
    ARTIFACT_TYPE,
    build_postprocessing_export,
    preview_postprocessing_export,
    verify_postprocessing_export,
)
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import confirm_coarse, set_coarse_draft, skip_fine
from ftir_workbench.batch.workspace import json_bytes, make_archive, read_verified_archive
from tests.batch.test_importing import table_bytes
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_state_contracts import apply_branch, graph


def _upload(workspace: BatchWorkspace, filename: str, labels: tuple[str, ...]) -> list[str]:
    x = np.linspace(1800.0, 900.0, 181)
    ys = [0.2 + (index + 1) * np.exp(-((x - 1450) / 40) ** 2) for index in range(len(labels))]
    ids = import_sources(workspace, [(filename, table_bytes(x, *ys, labels=labels))])
    for sid in ids:
        set_coarse_draft(workspace, sid, CoarseBaselineConfig(method="none"))
        preview_coarse(workspace, sid)
        confirm_coarse(workspace, sid)
        skip_fine(workspace, sid)
    return ids


def _unpack(payload: bytes) -> tuple[dict[str, bytes], dict, dict, dict]:
    members = read_verified_archive(payload, ARTIFACT_TYPE)
    metadata = json.loads(members["batch_metadata.json"])
    index = json.loads(members["branch_index.json"])
    nodes = {(entry["spectrum_id"], entry["branch"]): json.loads(members[entry["recipe_json"]])
             for entry in index["nodes"]}
    return members, metadata, index, nodes


def test_requested_ns_preview_matches_readable_graph_and_wide_with_exact_parent_arrays() -> None:
    ws = BatchWorkspace()
    sid = _upload(ws, "G2_90RH.csv", ("Absorbance",))[0]
    graph(ws, sid)
    before = {branch: get_branch(ws, sid, branch) for branch in BRANCHES}
    preview = preview_postprocessing_export(ws, [sid], ("normalized_smoothed",))
    artifact = build_postprocessing_export(
        ws, [sid], branches=("normalized_smoothed",), include_wide=True,
    )
    members, metadata, index, nodes = _unpack(artifact.zip_bytes)
    assert metadata["schema_version"] == index["schema_version"] == "1.1"
    assert metadata["naming_policy_version"] == index["naming_policy_version"] == "1.0"
    assert all(node["schema_version"] == "1.0" for node in nodes.values())
    assert metadata["name_map"] == preview["name_map"]
    assert members["export_name_map.csv"] == _table(*name_map_table(metadata["name_map"]))
    rows = {row["branch"]: row for row in metadata["name_map"]}
    expected = {
        "baseline": "G2_90RH__baseline",
        "smoothed": "G2_90RH__baseline__smooth-gaussian",
        "normalized_smoothed": "G2_90RH__baseline__smooth-gaussian__norm-max",
    }
    for branch, label in expected.items():
        row, node = rows[branch], nodes[(sid, branch)]
        assert row["spectra_csv"] == node["spectra_csv"] == f"spectra/{label}.csv"
        assert row["wide_column_label"] == label and sid not in row["spectra_csv"]
        assert row["included_as"] == ("requested" if branch == "normalized_smoothed" else "parent")
        assert row["name_basis"] == "original_filename" and row["source_layout"] == "two_column"
        assert row["fingerprint"] == node["fingerprint"] == before[branch].fingerprint
        actual = np.loadtxt(io.BytesIO(members[node["spectra_csv"]]), delimiter=",", skiprows=1)
        snapshot = before[branch]
        if branch == "baseline":
            x, y = snapshot.result.absorbance_selected.wavenumber, snapshot.result.baseline.corrected
        else:
            x, y = snapshot.wavenumber, snapshot.spectra
        np.testing.assert_array_equal(actual[:, 0], x)
        np.testing.assert_array_equal(actual[:, 1], y[0])
    assert rows["smoothed"]["removed_csv"] == "components/G2_90RH__baseline__smooth-gaussian__removed_component.csv"
    assert metadata["wide_headers"] == [expected["normalized_smoothed"]]
    assert verify_postprocessing_export(artifact.zip_bytes)
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)
    assert all(get_branch(ws, sid, branch) is snapshot for branch, snapshot in before.items())


@pytest.mark.parametrize("method", ["savgol", "gaussian", "moving_average", "median"])
def test_names_use_confirmed_smoothing_and_ns_parent_not_unconfirmed_drafts(method: str) -> None:
    ws, sid = confirmed_workspace(name="sample.csv")
    apply_branch(ws, sid, "smoothed", {"enabled": True, "method": method})
    apply_branch(ws, sid, "normalized_smoothed", {"enabled": True, "method": "vector"})
    update_draft(ws, sid, "smoothed", {"method": "median" if method != "median" else "gaussian"})
    update_draft(ws, sid, "normalized_smoothed", {"method": "maximum"})
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_smoothed",))
    paths = {row["branch"]: row["spectra_csv"] for row in artifact.metadata["name_map"]}
    assert paths["smoothed"] == f"spectra/sample__baseline__smooth-{method}.csv"
    assert paths["normalized_smoothed"] == f"spectra/sample__baseline__smooth-{method}__norm-l2.csv"
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)


@pytest.mark.parametrize(("method", "suffix"), [
    ("maximum", "norm-max"), ("internal_peak_height", "norm-peak-window"),
    ("internal_peak_area", "norm-area-window"), ("area", "norm-area"),
    ("vector", "norm-l2"), ("minmax_display", "minmax_scaled_intensity_display_only"),
])
def test_confirmed_normalization_method_suffix_preserves_quantity(method: str, suffix: str) -> None:
    ws, sid = confirmed_workspace(name="样品 α (25°C).csv")
    apply_branch(ws, sid, "normalized_baseline", {
        "enabled": True, "method": method, "reference_interval": [1600.0, 1200.0],
    })
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_baseline",))
    members, metadata, _, nodes = _unpack(artifact.zip_bytes)
    row = next(row for row in metadata["name_map"] if row["branch"] == "normalized_baseline")
    node = nodes[(sid, "normalized_baseline")]
    assert row["spectra_csv"] == f"spectra/样品 α (25°C)__baseline__{suffix}.csv"
    assert row["quantity"] == node["quantity"]
    assert row["purpose"] == ("display_only" if method == "minmax_display" else "scientific")
    assert "absorbance" not in members[row["spectra_csv"]].decode().splitlines()[0]
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_wide_column_subset_and_collision_names_use_whole_workspace() -> None:
    ws = BatchWorkspace()
    a, b = _upload(ws, "wide.csv", ("Sample", "sample"))
    c = _upload(ws, "single.csv", ("Absorbance",))[0]
    together = build_postprocessing_export(ws, [a, b, c], include_wide=True)
    rows = {row["spectrum_id"]: row for row in together.metadata["name_map"]}
    assert rows[a]["name_basis"] == rows[b]["name_basis"] == "original_column_label"
    assert rows[a]["collision_token"] and rows[b]["collision_token"]
    assert rows[a]["spectra_csv"].casefold() != rows[b]["spectra_csv"].casefold()
    single = build_postprocessing_export(ws, [a])
    reverse = build_postprocessing_export(ws, [c, b, a], include_wide=True)
    assert single.metadata["name_map"][0] == rows[a]
    assert {row["spectrum_id"]: row for row in reverse.metadata["name_map"]} == rows
    assert together.metadata["wide_headers"] == [rows[sid]["wide_column_label"] for sid in [a, b, c]]
    assert verify_postprocessing_export(single.zip_bytes, recompute=False)


def test_display_override_map_escapes_csv_but_not_original_json() -> None:
    ws, sid = confirmed_workspace(name="original.csv")
    ws.records[sid] = replace(ws.records[sid], display_name="display α")
    display = build_postprocessing_export(ws, [sid], naming="display")
    assert display.metadata["name_map"][0]["spectra_csv"] == "spectra/display α__baseline.csv"
    override = "=curve (25°C)"
    artifact = build_postprocessing_export(
        ws, [sid], naming={"policy": "display", "overrides": {sid: override}}, include_wide=True,
    )
    members, metadata, _, _ = _unpack(artifact.zip_bytes)
    row = metadata["name_map"][0]
    csv_row = next(csv.DictReader(io.StringIO(members["export_name_map.csv"].decode())))
    assert row["override"] == override and row["name_basis"] == "override"
    assert csv_row["override"] == "'" + override
    assert row["original_filename"] == "original.csv" and row["display_name"] == "display α"
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)


@pytest.mark.parametrize("policy", ["original", "display"])
def test_explicit_null_override_clears_only_this_export_without_restoring_saved_value(policy: str) -> None:
    ws, sid = confirmed_workspace(name="original.csv")
    ws.records[sid] = replace(ws.records[sid], display_name="display name")
    graph(ws, sid)
    set_naming_policy(ws, policy)
    set_name_override(ws, sid, "saved override")
    before = get_naming_options(ws)
    naming = {"overrides": {sid: None}}
    preview = preview_postprocessing_export(ws, [sid], ("normalized_smoothed",), naming=naming)
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_smoothed",), naming=naming)
    expected = "original" if policy == "original" else "display name"
    assert preview["name_map"] == artifact.metadata["name_map"]
    assert all(row["override"] is None and row["base_name"] == expected for row in preview["name_map"])
    baseline = build_batch_export(ws, [sid], stage="fine", naming=naming)
    assert baseline.metadata["name_map"][0]["base_name"] == expected
    assert get_naming_options(ws) == before
    assert build_postprocessing_export(ws, [sid]).metadata["name_map"][0]["base_name"] == "saved override"
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)


@pytest.mark.parametrize("state", ["missing", "stale", "excluded", "empty_selection"])
def test_preview_reports_unavailable_results_without_packaging_or_fitting(state: str) -> None:
    ws, sid = confirmed_workspace()
    if state in {"stale", "excluded"}:
        graph(ws, sid)
    if state == "stale":
        apply_branch(ws, sid, "smoothed", {"enabled": True, "method": "median"})
    if state == "excluded":
        ws.records[sid] = replace(ws.records[sid], excluded=True)
    ids = [] if state == "empty_selection" else [sid]
    preview = preview_postprocessing_export(ws, ids, ("normalized_smoothed",))
    assert preview["name_map"] == []
    assert [row["status"] for row in preview["report"]] == ([] if not ids else [state])
    with pytest.raises(BatchError, match="EXPORT_NOT_READY"):
        build_postprocessing_export(ws, ids, branches=("normalized_smoothed",))


def test_legacy_preview_uses_actual_old_paths_without_changing_legacy_members() -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    preview = preview_postprocessing_export(ws, [sid], BRANCHES, naming="legacy_id")
    artifact = build_postprocessing_export(ws, [sid], branches=BRANCHES, include_wide=True, naming="legacy_id")
    members, metadata, index, nodes = _unpack(artifact.zip_bytes)
    assert metadata["schema_version"] == index["schema_version"] == "1.0"
    assert "name_map" not in metadata and "naming_policy_version" not in index
    assert "export_name_map.csv" not in members
    assert [row["wide_column_label"] for row in preview["name_map"]] == metadata["wide_headers"]
    for row in preview["name_map"]:
        assert row["spectra_csv"] == nodes[(sid, row["branch"])]["spectra_csv"]
        assert sid in row["spectra_csv"]
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)


@pytest.mark.parametrize("attack", [
    "source_id", "source_column", "quantity", "role", "parent_hash", "base_name",
    "collision_token", "unknown_field", "omit_parent_map", "duplicate_map", "wrong_suffix",
    "map_csv", "extra_member", "mixed_schema", "unknown_schema", "node_schema",
    "policy_version", "policy_mismatch", "recipe_path", "inconsistent_per_spectrum_decision",
    "inconsistent_layout",
])
def test_rehashed_readable_name_mapping_and_member_attacks_fail_closed(attack: str) -> None:
    ws, sid = confirmed_workspace(name="sample.csv")
    graph(ws, sid)
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_smoothed",), include_wide=True)
    members, metadata, index, nodes = _unpack(artifact.zip_bytes)
    row = metadata["name_map"][-1]
    if attack == "source_id":
        row["source_id"] = "another-source"
    elif attack == "source_column":
        row["source_column_index"] += 1
    elif attack == "quantity":
        row["quantity"] = "absorbance"
    elif attack == "role":
        row["included_as"] = "parent"
    elif attack == "parent_hash":
        row["parent_fingerprint"] = "a" * 64
    elif attack == "base_name":
        row["base_name"] = "different"
    elif attack == "collision_token":
        row["collision_token"] = "12345678"
    elif attack == "unknown_field":
        row["unrecognized"] = "ignored?"
    elif attack == "omit_parent_map":
        metadata["name_map"].pop(0)
    elif attack == "duplicate_map":
        metadata["name_map"].append(dict(row))
    elif attack in {"wrong_suffix", "recipe_path"}:
        field = "spectra_csv" if attack == "wrong_suffix" else "recipe_json"
        old = row[field]
        new = "spectra/sample__baseline__smooth-median__norm-max.csv" if attack == "wrong_suffix" else "recipes/different.json"
        row[field] = new
        nodes[(sid, "normalized_smoothed")][field] = new
        index["nodes"][-1][field] = new
        members[new] = members.pop(old)
    elif attack == "extra_member":
        members["unreferenced.csv"] = b"extra\n"
    elif attack == "mixed_schema":
        index["schema_version"] = "1.0"
    elif attack == "unknown_schema":
        metadata["schema_version"] = index["schema_version"] = "9.0"
    elif attack == "node_schema":
        nodes[(sid, "normalized_smoothed")]["schema_version"] = "1.1"
    elif attack == "policy_version":
        metadata["naming_policy_version"] = index["naming_policy_version"] = "9.0"
    elif attack == "policy_mismatch":
        metadata["naming_policy"] = "display"
    elif attack == "inconsistent_per_spectrum_decision":
        row.update(override="different", raw_base_name="different", base_name="different", name_basis="override")
        node = nodes[(sid, "normalized_smoothed")]
        old_path = row["spectra_csv"]
        replacement = make_name_row(
            row, branch=node["branch"], quantity=node["quantity"], purpose=node["purpose"],
            fingerprint=node["fingerprint"], parent_fingerprint=node["parent_fingerprint"],
            recipe_json=node["recipe_json"], included_as="requested",
            effective_recipe=node["effective_recipe"],
            parent_effective_recipe=nodes[(sid, "smoothed")]["effective_recipe"],
        )
        metadata["name_map"][-1] = replacement
        node["spectra_csv"] = index["nodes"][-1]["spectra_csv"] = replacement["spectra_csv"]
        members[replacement["spectra_csv"]] = members.pop(old_path)
        wide = np.loadtxt(io.BytesIO(members[metadata["wide_csv"]]), delimiter=",", skiprows=1)
        metadata["wide_headers"] = [replacement["wide_column_label"]]
        members[metadata["wide_csv"]] = _table(["wavenumber_cm-1", *metadata["wide_headers"]], wide.tolist())
    elif attack == "inconsistent_layout":
        for mapped in metadata["name_map"]:
            mapped["layout_evidence"]["probe"]["layout"] = "wide_table"
            mapped["source_layout"] = "wide_table"
            mapped["name_basis"] = "original_column_label"
    for node in nodes.values():
        node["node_integrity_sha256"] = json_fingerprint({key: value for key, value in node.items() if key != "node_integrity_sha256"})
        members[node["recipe_json"]] = json_bytes(node)
    members["export_name_map.csv"] = _table(*name_map_table(metadata["name_map"]))
    if attack == "map_csv":
        members["export_name_map.csv"] += b"invented,row\n"
    members["batch_metadata.json"] = json_bytes(metadata)
    members["branch_index.json"] = json_bytes(index)
    assert not verify_postprocessing_export(make_archive(members, ARTIFACT_TYPE), recompute=False)


@pytest.mark.parametrize("exporter", ["baseline", "postprocessing"])
@pytest.mark.parametrize("attack", ["contradictory_layout", "insufficient_source_count"])
def test_rehashed_same_source_columns_must_share_consistent_naming_evidence(
    exporter: str, attack: str,
) -> None:
    ws = BatchWorkspace()
    first, second = _upload(ws, "wide.csv", ("columnA", "columnB"))
    baseline = exporter == "baseline"
    artifact_type = "independent_baseline_batch" if baseline else ARTIFACT_TYPE
    artifact = (build_batch_export(ws, [first, second], stage="fine") if baseline
                else build_postprocessing_export(ws, [first, second]))
    members = read_verified_archive(artifact.zip_bytes, artifact_type)
    metadata = json.loads(members["batch_metadata.json"])
    rows = metadata["name_map"]
    assert rows[0]["source_id"] == rows[1]["source_id"]
    assert rows[0]["layout_evidence"] == rows[1]["layout_evidence"]
    if attack == "insufficient_source_count":
        # Each row remains individually plausible, but the package contains two
        # different source columns rather than the claimed one workspace entry.
        for row in rows:
            row["layout_evidence"]["source_record_count"] = 1
    else:
        # Keep each row internally valid while claiming the exact same source
        # was a two-column file for A and a wide table for B. Rename actual
        # members and every reference too, so path/hash checks alone cannot help.
        row = rows[0]
        previous_spectra, previous_components = row["spectra_csv"], row["components_csv"]
        row.update(
            source_layout="two_column",
            layout_evidence={
                "probe": {"layout": "two_column", "columns": 2, "header_present": True},
                "source_record_count": 1, "max_column_index": 1,
            },
            name_basis="original_filename", raw_base_name="wide", base_name="wide",
        )
        replacement = make_name_row(
            row, branch="baseline", quantity="absorbance", purpose="scientific",
            fingerprint=row["fingerprint"], parent_fingerprint=row["parent_fingerprint"],
            recipe_json=row["recipe_json"], included_as="requested", components=baseline,
        )
        rows[0] = replacement
        members[replacement["spectra_csv"]] = members.pop(previous_spectra)
        recipe = json.loads(members[row["recipe_json"]])
        if baseline:
            members[replacement["components_csv"]] = members.pop(previous_components)
            entry = metadata["spectra"][0]
            entry.update(spectra_csv=replacement["spectra_csv"],
                         components_csv=replacement["components_csv"])
            recipe["spectrum"] = entry
        else:
            index = json.loads(members["branch_index.json"])
            index["nodes"][0]["spectra_csv"] = replacement["spectra_csv"]
            recipe["spectra_csv"] = replacement["spectra_csv"]
            recipe["node_integrity_sha256"] = json_fingerprint({
                key: value for key, value in recipe.items() if key != "node_integrity_sha256"
            })
            members["branch_index.json"] = json_bytes(index)
        members[row["recipe_json"]] = json_bytes(recipe)
    members["batch_metadata.json"] = json_bytes(metadata)
    members["export_name_map.csv"] = _table(*name_map_table(rows))
    tampered = make_archive(members, artifact_type)
    accepted = (verify_batch_export(tampered) if baseline
                else verify_postprocessing_export(tampered, recompute=False))
    assert not accepted


def test_naming_preview_export_and_changed_options_do_not_call_scientific_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    ws, sid = confirmed_workspace(name="before.csv")
    graph(ws, sid)
    snapshots = {branch: get_branch(ws, sid, branch) for branch in BRANCHES}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("naming and serialization must not execute scientific processing")

    for target in (
        "ftir_workbench.batch.service.run_pipeline",
        "ftir_baseline.pipeline.run_pipeline",
        "ftir_workbench.batch.postprocessing.smooth_spectral_arrays",
        "ftir_workbench.batch.postprocessing.normalize_spectral_arrays",
        "ftir_workbench.batch.postprocessing_export.smooth_spectral_arrays",
        "ftir_workbench.batch.postprocessing_export.normalize_spectral_arrays",
        "ftir_workbench.models.PreparedSpectralDataset.__init__",
    ):
        monkeypatch.setattr(target, forbidden)
    first = build_postprocessing_export(ws, [sid], branches=BRANCHES)
    payload = first.zip_bytes
    for naming in ("original", "display", "legacy_id", {"overrides": {sid: "override α"}}):
        preview_postprocessing_export(ws, [sid], BRANCHES, naming=naming)
        result = build_postprocessing_export(ws, [sid], branches=BRANCHES, naming=naming)
        assert verify_postprocessing_export(result.zip_bytes, recompute=False)
    assert first.zip_bytes == payload
    assert all(get_branch(ws, sid, branch) is snapshot for branch, snapshot in snapshots.items())
