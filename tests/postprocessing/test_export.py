"""PP-023/042/055-061: actual bytes, confirmed parent graphs and separate replay."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import ftir_workbench.batch.postprocessing_export as export_module
from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch.export import build_batch_export
from ftir_workbench.batch.models import BatchError
from ftir_workbench.batch.postprocessing import (
    BRANCHES,
    branch_state,
    get_branch,
    update_draft,
)
from ftir_workbench.batch.postprocessing_export import (
    ARTIFACT_TYPE,
    build_postprocessing_export,
    can_export_postprocessing_wide,
    verify_postprocessing_export,
)
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import confirm_coarse, set_coarse_draft
from ftir_workbench.batch.workspace import make_archive, read_verified_archive
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_state_contracts import apply_branch, graph


def unpack(artifact: Any) -> tuple[dict[str, bytes], dict[tuple[str, str], dict[str, Any]]]:
    members = read_verified_archive(artifact.zip_bytes, ARTIFACT_TYPE)
    index = json.loads(members["branch_index.json"])
    return members, {
        (entry["spectrum_id"], entry["branch"]): json.loads(members[entry["recipe_json"]])
        for entry in index["nodes"]
    }


def values(members: dict[str, bytes], node: dict[str, Any]) -> np.ndarray:
    return np.loadtxt(io.BytesIO(members[node["spectra_csv"]]), delimiter=",", skiprows=1, ndmin=2)


@pytest.mark.parametrize(
    ("branch", "expected_graph"),
    [
        ("baseline", {"baseline"}),
        ("smoothed", {"baseline", "smoothed"}),
        ("normalized_baseline", {"baseline", "normalized_baseline"}),
        ("normalized_smoothed", {"baseline", "smoothed", "normalized_smoothed"}),
    ],
)
def test_pp059_requested_branch_includes_exact_complete_parent_graph(
    branch: str,
    expected_graph: set[str],
) -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    artifact = build_postprocessing_export(ws, [sid], branches=(branch,))
    assert verify_postprocessing_export(artifact.zip_bytes)
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)
    members, nodes = unpack(artifact)
    assert {key[1] for key in nodes} == expected_graph
    assert artifact.metadata["is_2d_ready"] is False
    assert artifact.metadata["requested_branches"] == [branch]
    assert artifact.metadata["exported_items"][0]["branch"] == branch
    assert all("for_2dcos" not in name and "prepared" not in name.lower() for name in members)
    assert all(not name.startswith("sources/") for name in members)
    for (_, kind), node in nodes.items():
        snapshot = get_branch(ws, sid, kind)
        x, y = export_module.branch_arrays(snapshot)
        data = values(members, node)
        np.testing.assert_array_equal(data[:, 0], x)
        np.testing.assert_array_equal(data[:, 1], y[0])
        assert node["fingerprint"] == snapshot.fingerprint
        assert node["source"]["source_id"] == ws.records[sid].source_id
        assert node["source"]["source_column_index"] == ws.records[sid].original_column_index
        assert node["is_2d_ready"] is False
        if kind != "baseline":
            assert node["recipe"] == export_module.plain(snapshot.recipe)
            assert node["effective_recipe"] == export_module.plain(snapshot.effective_recipe)
            assert node["parent_fingerprint"] == snapshot.parent_fingerprint
            assert node["versions"] == dict(snapshot.versions)
    if "smoothed" in expected_graph:
        smoothed = nodes[(sid, "smoothed")]
        residual = np.loadtxt(
            io.BytesIO(members[smoothed["removed_csv"]]), delimiter=",", skiprows=1
        )
        np.testing.assert_array_equal(
            residual[:, 1], get_branch(ws, sid, "smoothed").removed_component[0]
        )
        assert (
            "smoothing_removed_component"
            in members[smoothed["removed_csv"]].decode().splitlines()[0]
        )


def test_pp055_export_and_nonreplay_verifier_never_call_numerical_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    parents = {branch: get_branch(ws, sid, branch) for branch in BRANCHES}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("export must serialize confirmed arrays without numerical processing")

    for target in (
        "ftir_baseline.pipeline.run_pipeline",
        "ftir_workbench.batch.service.run_pipeline",
        "ftir_workbench.models.PreparedSpectralDataset.__init__",
        "ftir_workbench.batch.postprocessing.smooth_spectral_arrays",
        "ftir_workbench.batch.postprocessing.normalize_spectral_arrays",
        "ftir_workbench.post_baseline_smoothing.smooth_spectral_arrays",
        "ftir_workbench.batch.normalization_adapter.apply_normalization",
        "ftir_workbench.batch.normalization_adapter.normalize_spectral_arrays",
        "ftir_workbench.batch.postprocessing_export.smooth_spectral_arrays",
        "ftir_workbench.batch.postprocessing_export.normalize_spectral_arrays",
    ):
        monkeypatch.setattr(target, forbidden)
    assert can_export_postprocessing_wide(ws, [sid], branches=BRANCHES) == (True, "")
    artifact = build_postprocessing_export(ws, [sid], branches=BRANCHES, include_wide=True)
    assert verify_postprocessing_export(artifact.zip_bytes, recompute=False)
    assert ws.export_history == [] and ws.last_export_summary is None
    for branch, parent in parents.items():
        assert get_branch(ws, sid, branch) is parent


def test_replay_verifier_explicitly_replays_only_smoothed_and_normalized_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    artifact = build_postprocessing_export(ws, [sid], branches=BRANCHES)
    calls = {"smooth": 0, "normalize": 0}
    smooth, normalize = (
        export_module.smooth_spectral_arrays,
        export_module.normalize_spectral_arrays,
    )

    def smoothing(*args: Any, **kwargs: Any) -> Any:
        calls["smooth"] += 1
        return smooth(*args, **kwargs)

    def normalization(*args: Any, **kwargs: Any) -> Any:
        calls["normalize"] += 1
        return normalize(*args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("even verification cannot refit baseline or construct Prepared")

    monkeypatch.setattr(export_module, "smooth_spectral_arrays", smoothing)
    monkeypatch.setattr(export_module, "normalize_spectral_arrays", normalization)
    monkeypatch.setattr("ftir_baseline.pipeline.run_pipeline", forbidden)
    monkeypatch.setattr("ftir_workbench.batch.service.run_pipeline", forbidden)
    monkeypatch.setattr("ftir_workbench.models.PreparedSpectralDataset.__init__", forbidden)
    assert verify_postprocessing_export(artifact.zip_bytes)
    assert calls == {"smooth": 1, "normalize": 2}


def test_pp023_csv_reads_corrected_normalized_output_not_parent_analysis_data() -> None:
    ws, sid = confirmed_workspace(np.array([1000.0, 1002, 1004]), np.array([0.0, 2, 4]))
    apply_branch(ws, sid, "normalized_baseline", {"enabled": True, "method": "maximum"})
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_baseline",))
    members, nodes = unpack(artifact)
    np.testing.assert_array_equal(values(members, nodes[(sid, "baseline")])[:, 1], [0, 2, 4])
    np.testing.assert_array_equal(
        values(members, nodes[(sid, "normalized_baseline")])[:, 1], [0, 0.5, 1]
    )
    assert verify_postprocessing_export(artifact.zip_bytes)


@pytest.mark.parametrize("source", ["baseline", "smoothed"])
def test_pp058_minmax_export_quantity_offset_and_filename_are_explicit(source: str) -> None:
    ws, sid = confirmed_workspace(np.arange(9.0), np.linspace(-1, 3, 9))
    if source == "smoothed":
        apply_branch(ws, sid, "smoothed", {"enabled": True, "method": "moving_average"})
    branch = "normalized_" + source
    expected = apply_branch(ws, sid, branch, {"enabled": True, "method": "minmax_display"})
    artifact = build_postprocessing_export(ws, [sid], branches=(branch,), include_wide=True)
    members, nodes = unpack(artifact)
    node = nodes[(sid, branch)]
    assert node["purpose"] == "display_only"
    assert node["quantity"] == "minmax_scaled_intensity"
    assert "minmax_scaled_intensity_display_only" in node["spectra_csv"]
    assert (
        members[node["spectra_csv"]].decode().splitlines()[0]
        == "wavenumber_cm-1,minmax_scaled_intensity_display_only"
    )
    np.testing.assert_array_equal(values(members, node)[:, 1], expected.spectra[0])
    np.testing.assert_array_equal(node["scale"], expected.scale)
    np.testing.assert_array_equal(node["offset"], expected.offset)
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_pp058_float64_tiny_negative_and_ordinary_values_roundtrip_exactly() -> None:
    y = np.array([-0.12345678901234567, 2e-100, 1.2345678901234567, -3e-200, 0.25])
    ws, sid = confirmed_workspace(np.arange(5.0), y)
    expected = apply_branch(
        ws, sid, "normalized_baseline", {"enabled": True, "method": "maximum", "target": 1e-100}
    )
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_baseline",))
    members, nodes = unpack(artifact)
    np.testing.assert_array_equal(
        values(members, nodes[(sid, "normalized_baseline")])[:, 1], expected.spectra[0]
    )
    np.testing.assert_array_equal(values(members, nodes[(sid, "baseline")])[:, 1], y)
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_pp056_missing_branch_default_blocks_and_valid_only_explicitly_reports_no_fallback() -> (
    None
):
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="only-B.csv")
    graph(ws, a)
    for valid_only in (False,):
        with pytest.raises(BatchError, match="EXPORT_NOT_READY"):
            build_postprocessing_export(
                ws, [a, b], branches=("normalized_smoothed",), valid_only=valid_only
            )
    artifact = build_postprocessing_export(
        ws, [a, b], branches=("normalized_smoothed",), valid_only=True
    )
    assert [(row["spectrum_id"], row["branch"]) for row in artifact.metadata["exported_items"]] == [
        (a, "normalized_smoothed")
    ]
    assert artifact.report[1]["spectrum_id"] == b
    assert artifact.report[1]["status"] == "missing"
    assert artifact.report[1]["reason_code"] == "POSTPROCESS_REQUIRED"
    _, nodes = unpack(artifact)
    assert all(sid == a for sid, _ in nodes)
    assert verify_postprocessing_export(artifact.zip_bytes)
    all_available = build_postprocessing_export(ws, [a, b], branches=BRANCHES, valid_only=True)
    assert len(all_available.report) == 8
    assert len(all_available.metadata["exported_items"]) == 5
    assert verify_postprocessing_export(all_available.zip_bytes)


def test_pp056_stale_smoothing_child_and_stale_final_baseline_are_not_exported() -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    apply_branch(ws, sid, "smoothed", {"gaussian_sigma_points": 2.0})
    with pytest.raises(BatchError, match="POSTPROCESS_STALE"):
        build_postprocessing_export(ws, [sid], branches=("normalized_smoothed",))
    current = build_postprocessing_export(ws, [sid], branches=BRANCHES, valid_only=True)
    assert (
        next(row for row in current.report if row["branch"] == "normalized_smoothed")["status"]
        == "stale"
    )
    assert verify_postprocessing_export(current.zip_bytes)
    set_coarse_draft(ws, sid, CoarseBaselineConfig(method="offset"))
    preview_coarse(ws, sid)
    confirm_coarse(ws, sid)
    with pytest.raises(BatchError, match="FINE_STALE"):
        build_postprocessing_export(ws, [sid], branches=BRANCHES)


def test_reports_unconfirmed_edits_failures_exclusions_and_import_issues_without_applying_them() -> (
    None
):
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="failed-N.csv")
    _, c = confirmed_workspace(workspace=ws, name="excluded.csv")
    original = apply_branch(ws, a, "normalized_baseline", {"enabled": True, "method": "maximum"})
    update_draft(ws, a, "normalized_baseline", {"target": -1})
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="offset"))
    ws.states[a].display_preferences["error_fine"] = "unfinished anchors"
    branch_state(ws, b, "normalized_baseline").errors[:] = [
        "NORMALIZATION_REFERENCE_TOO_SMALL: synthetic"
    ]
    ws.records[c] = replace(ws.records[c], excluded=True)
    ws.import_issues.append({"code": "IMPORT_FAILED", "original_filename": "bad.csv"})
    artifact = build_postprocessing_export(
        ws, [a, b, c], branches=("normalized_baseline",), valid_only=True
    )
    assert [row["status"] for row in artifact.report] == ["exported", "failed", "excluded"]
    assert artifact.report[0]["draft_modified"]
    assert artifact.report[0]["invalid_editor_not_used"]["error_fine"] == "unfinished anchors"
    assert "synthetic" in artifact.report[1]["recorded_errors"][0]
    assert artifact.metadata["import_issues"][0]["code"] == "IMPORT_FAILED"
    assert get_branch(ws, a, "normalized_baseline") is original
    assert verify_postprocessing_export(artifact.zip_bytes)


@pytest.mark.parametrize("axis_kind", ["identical", "coordinate", "reverse", "short", "range"])
def test_pp057_only_exact_actual_x_arrays_allow_wide(axis_kind: str) -> None:
    x, y = np.arange(9.0), np.arange(9.0) + 1
    ws, a = confirmed_workspace(x, y)
    other_x = x.copy()
    if axis_kind == "coordinate":
        other_x[2] += 0.01
    elif axis_kind == "reverse":
        other_x = other_x[::-1]
    elif axis_kind == "short":
        other_x = other_x[:-1]
    elif axis_kind == "range":
        other_x = other_x + 100
    _, b = confirmed_workspace(other_x, np.arange(len(other_x)) + 2.0, workspace=ws)
    for sid in (a, b):
        apply_branch(ws, sid, "normalized_baseline", {"enabled": True, "method": "maximum"})
    available, reason = can_export_postprocessing_wide(
        ws, [a, b], branches=("normalized_baseline",)
    )
    assert available is (axis_kind == "identical")
    if available:
        artifact = build_postprocessing_export(
            ws, [a, b], branches=("normalized_baseline",), include_wide=True
        )
        assert artifact.wide_csv_bytes is not None
    else:
        assert reason
        with pytest.raises(BatchError, match="AXIS_MISMATCH_FOR_WIDE_EXPORT"):
            build_postprocessing_export(
                ws, [a, b], branches=("normalized_baseline",), include_wide=True
            )
        artifact = build_postprocessing_export(ws, [a, b], branches=("normalized_baseline",))
        assert artifact.wide_csv_bytes is None and artifact.wide_unavailable_reason
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_pp042_heterogeneous_recipes_warn_even_when_output_axes_match() -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws)
    apply_branch(ws, a, "normalized_baseline", {"enabled": True, "method": "maximum"})
    apply_branch(
        ws,
        b,
        "normalized_baseline",
        {"enabled": True, "method": "area", "area_definition": "absolute"},
    )
    artifact = build_postprocessing_export(
        ws, [a, b], branches=("normalized_baseline",), include_wide=True
    )
    assert any(
        "Heterogeneous normalized_baseline" in warning for warning in artifact.metadata["warnings"]
    )
    assert any("quantitative comparability" in warning for warning in artifact.metadata["warnings"])
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_pp060_rename_and_edit_do_not_mutate_existing_payload_or_refit() -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    first = build_postprocessing_export(ws, [sid], branches=BRANCHES, include_wide=True)
    original_bytes = first.zip_bytes
    originals = {branch: get_branch(ws, sid, branch).fingerprint for branch in BRANCHES}
    ws.records[sid] = replace(ws.records[sid], display_name="a new label")
    update_draft(ws, sid, "smoothed", {"gaussian_sigma_points": 5.0})
    second = build_postprocessing_export(ws, [sid], branches=BRANCHES, include_wide=True)
    assert first.zip_bytes == original_bytes
    assert second.zip_bytes != original_bytes
    assert second.report[1]["draft_modified"]
    assert second.report[0]["display_name"] == "a new label"
    assert all(get_branch(ws, sid, branch).fingerprint == originals[branch] for branch in BRANCHES)
    assert verify_postprocessing_export(first.zip_bytes) and verify_postprocessing_export(
        second.zip_bytes
    )


@pytest.mark.parametrize("name", ["=1+1", "+SUM(1)", "-formula", "@link", "  =1", "\t=1", "\rtext"])
def test_formula_like_labels_are_csv_escaped_but_metadata_and_numbers_are_preserved(
    name: str,
) -> None:
    ws, sid = confirmed_workspace(np.arange(5.0), np.array([-0.3, 0.1, 0.2, 0.1, 0.3]))
    ws.records[sid] = replace(ws.records[sid], display_name=name)
    apply_branch(ws, sid, "normalized_baseline", {"enabled": True, "method": "maximum"})
    artifact = build_postprocessing_export(
        ws, [sid], branches=("normalized_baseline",), include_wide=True
    )
    members, nodes = unpack(artifact)
    report = list(csv.DictReader(io.StringIO(members["processing_report.csv"].decode())))
    assert report[0]["display_name"] == "'" + name
    assert nodes[(sid, "normalized_baseline")]["source"]["display_name"] == name
    assert values(members, nodes[(sid, "normalized_baseline")])[0, 1] < 0
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_old_baseline_export_bytes_remain_unchanged_after_postprocessing_and_new_export() -> None:
    ws, sid = confirmed_workspace()
    before = build_batch_export(ws, [sid], stage="fine", include_wide=True)
    graph(ws, sid)
    artifact = build_postprocessing_export(ws, [sid], branches=BRANCHES, include_wide=True)
    after = build_batch_export(ws, [sid], stage="fine", include_wide=True)
    assert after.zip_bytes == before.zip_bytes
    assert artifact.metadata["artifact_type"] == ARTIFACT_TYPE
    assert verify_postprocessing_export(artifact.zip_bytes)


def test_baseline_only_versions_distinguish_saved_core_from_export_environment() -> None:
    ws, sid = confirmed_workspace()
    parent = get_branch(ws, sid, "baseline")
    artifact = build_postprocessing_export(ws, [sid])
    _, nodes = unpack(artifact)
    baseline = nodes[(sid, "baseline")]
    assert baseline["versions"] == {"ftir_baseline": parent.result.software_version}
    assert "did not separately store numerical dependency versions" in baseline["version_scope"]
    assert "not baseline calculation time" in artifact.metadata["baseline_version_scope"]
    assert set(artifact.metadata["exporter_versions"]) == {
        "python", "numpy", "scipy", "ftir-spectral-workbench", "pybaselines", "pydantic",
    }
    assert verify_postprocessing_export(artifact.zip_bytes)


@pytest.mark.parametrize(
    ("ids", "branches"),
    [
        ([], ("baseline",)),
        (["missing"], ("baseline",)),
        (["placeholder"], ()),
        (["placeholder"], ("coarse",)),
    ],
)
def test_empty_unknown_or_unsupported_selection_rejected(
    ids: list[str], branches: tuple[str, ...]
) -> None:
    ws, sid = confirmed_workspace()
    ids = [sid if value == "placeholder" else value for value in ids]
    with pytest.raises(BatchError, match="EXPORT_NOT_READY"):
        build_postprocessing_export(ws, ids, branches=branches)
    assert not can_export_postprocessing_wide(ws, ids, branches=branches)[0]


def test_pp061_truncated_or_rehashed_missing_member_bundle_is_rejected() -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    artifact = build_postprocessing_export(ws, [sid], branches=BRANCHES)
    assert not verify_postprocessing_export(artifact.zip_bytes[:50])
    members, nodes = unpack(artifact)
    members.pop(nodes[(sid, "baseline")]["spectra_csv"])
    assert not verify_postprocessing_export(make_archive(members, ARTIFACT_TYPE))


@pytest.mark.parametrize("method", ["savgol", "gaussian", "moving_average", "median"])
def test_all_filter_outputs_replay_from_exported_parent(method: str) -> None:
    ws, sid = confirmed_workspace()
    apply_branch(ws, sid, "smoothed", {"enabled": True, "method": method})
    artifact = build_postprocessing_export(ws, [sid], branches=("smoothed",))
    assert verify_postprocessing_export(artifact.zip_bytes)


@pytest.mark.parametrize(
    "method",
    ["maximum", "internal_peak_height", "internal_peak_area", "area", "vector", "minmax_display"],
)
def test_all_normalization_outputs_replay_from_exported_parent(method: str) -> None:
    ws, sid = confirmed_workspace()
    apply_branch(
        ws,
        sid,
        "normalized_baseline",
        {
            "enabled": True,
            "method": method,
            "reference_interval": [1700, 1400],
        },
    )
    artifact = build_postprocessing_export(ws, [sid], branches=("normalized_baseline",))
    assert verify_postprocessing_export(artifact.zip_bytes)
